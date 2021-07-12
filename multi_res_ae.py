from __future__ import division
import os
from glob import glob
import tensorflow as tf
import numpy as np
import ops
import scipy
import torch
import lpips
from image_utils import load_image, save_batch_images

class multi_res_ae(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=256,  # size the input images
                 num_input_channels=3,  # number of channels of input images
                 num_z_channels=512,  # number of channels of the layer z (noise or code)
                 is_training=True,  # flag for training or testing mode
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 dataset_name='dataset',  # name of the dataset in the folder ./data
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.num_input_channels = num_input_channels
        self.num_z_channels = num_z_channels
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.coarse_res_p = (8, 4)
        self.middle_res = (32, 8)

        # ************************************* input to graph ********************************************************
        self.input_image = tf.placeholder(
            tf.float32,
            [None, self.num_input_channels, self.size_image, self.size_image],
            name='input_images'
        ) 
        
        self.input_image_mix = tf.placeholder(
            tf.float32,
            [None, self.num_input_channels, self.size_image, self.size_image],
            name='input_images_mix'
        )
        # ************************************* build the graph *******************************************************
        print('\n\tBuilding graph ...')
        
        num_layers = int(np.log2(self.size_image))*2 - 2

        self.w_x = self.encoder(
            image=self.input_image,
            edlatent_broadcast = num_layers
        )
        self.w_x_mix = self.encoder(
            image=self.input_image_mix,
            edlatent_broadcast = num_layers,
            reuse_variables=True
        )
        
        # mixing latents w at coarse scale and at middle scale
        layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
        self.w_xx_c = tf.where(tf.broadcast_to(layer_idx < self.coarse_res_p[1], tf.shape(self.w_x)), self.w_x, self.w_x_mix)
        self.w_xx_m = tf.where(tf.broadcast_to(layer_idx < self.middle_res[1], tf.shape(self.w_x)), self.w_x, self.w_x_mix)
        
        self.G = self.decoder(
            w = self.w_x
        )
        self.G_xx_c = self.decoder(
            w = self.w_xx_c,
            reuse_variables=True
        )
        self.G_xx_m = self.decoder(
            w = self.w_xx_m,
            reuse_variables=True
        )
        
        # Discriminator takes both real and generated images
        self.real_logits = self.discriminator_img(
                image = self.input_image, # tf.concat([self.input_image, self.input_image_mix], 0),
        )
        self.fake_logits = self.discriminator_img(
                image = self.G, # tf.concat([self.G, self.G_xx_c, self.G_xx_m], 0),
                reuse_variables=True
        )

        # mower the resolution of the generated images with  mixed latents
        self.G_xx_8 = ops.adapt_scale(self.G_xx_c, int(self.size_image/self.coarse_res_p[0]))
        self.input_image_8 = ops.adapt_scale(self.input_image, int(self.size_image/self.coarse_res_p[0])) 
        self.G_xx_32 = ops.adapt_scale(self.G_xx_m, int(self.size_image/self.middle_res[0]))
        self.input_image_32 = ops.adapt_scale(self.input_image, int(self.size_image/self.middle_res[0])) 
        
        # reconstruction losses
        # without mixing
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss
        # with mixing
        self.EG_loss_mix_c = tf.reduce_mean(tf.abs(self.input_image_8 - self.G_xx_8))  # L1 loss
        self.EG_loss_mix_m = tf.reduce_mean(tf.abs(self.input_image_32 - self.G_xx_32))  # L1 loss
        
        # adversarialloss
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits, labels=tf.ones_like(self.real_logits))
        )
        self.D_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits))
        )
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.ones_like(self.fake_logits))
        )
        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        # ************************************* collect the summary ***************************************
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.EG_loss_mix_c_summary = tf.summary.scalar('EG_loss_mix_c', self.EG_loss_mix_c)
        self.EG_loss_mix_m_summary = tf.summary.scalar('EG_loss_mix_m', self.EG_loss_mix_m)
        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self,
              num_epochs=150,  # number of epochs
              init_image_res = 8,
              beta1=0.5,  # parameter for Adam optimizer
              learning_rate=0.0001,  # learning rate of optimizer
              decay_rate=0.9,  # learning rate decay (0, 1], 1 means no decay
              decay_steps=10000,
              enable_shuffle=False,  # enable shuffle of the dataset
              use_trained_model=True,  # use the saved checkpoint to initialize the network
              gan_weight = 0.0005,
              sample_size = 12,
              ):

        # *************************** load file names of images ******************************************************
        file_names = glob(os.path.join('./', self.dataset_name, '*.jpg'))
        np.random.seed(seed=1000)
        if enable_shuffle:
            np.random.shuffle(file_names)

        # *********************************** optimizer **************************************************************
        self.loss_EG = self.EG_loss + self.EG_loss_mix_c + self.EG_loss_mix_m + gan_weight * (self.G_img_loss)
        self.loss_D = self.D_loss_real + self.D_loss_G

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate= learning_rate,
            global_step=self.EG_global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            # optimizer for autoencoder reconstruction loss
            self.EG_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_EG,
                global_step=self.EG_global_step,
                var_list=self.E_variables + self.G_variables
            )
            # optimizer for discriminator adversarial loss
            self.D_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_D,
                global_step=self.EG_global_step,
                var_list=self.D_img_variables
            )
        
        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.EG_loss_summary,
            self.EG_loss_mix_c_summary,
            self.EG_loss_mix_m_summary,
            self.EG_learning_rate_summary,
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        # ************* get some random samples as testing data to visualize the learning process *********************
        sample_files = file_names[0:sample_size*2]
        sample = [load_image(
            image_path=sample_file,
            lod=0.0,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
            
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            
        save_batch_images(
            batch_images=sample_images[:sample_size],
            save_path=os.path.join(sample_dir, 'reals1.png'),
            image_value_range=self.image_value_range,
            size_frame=[3,4]
        )
        save_batch_images(
            batch_images=sample_images[sample_size:],
            save_path=os.path.join(sample_dir, 'reals2.png'),
            image_value_range=self.image_value_range,
            size_frame=[3,4]
        )
        
        # ******************************************* training *******************************************************
        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")                    

        optimizer_vars = []
        for var in tf.global_variables():
            if var.name=="G_lod/G_lod:0":
                G_lod_var = var
            elif var.name=="E_lod/E_lod:0":
                E_lod_var = var
            elif var.name=="D_img_lod/D_img_lod:0":
                D_img_lod_var = var
            elif 'opt' in var.name:
                optimizer_vars.append(var)
        
        #----------------------------------------------------------------------
        initial_lod = int(np.log2(self.size_image)) - int(np.log2(init_image_res))
        lods = [l/10 for l in range(initial_lod*10, -1, -1)]        
        batch_sizes = {8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
        # epoch iteration
        prev_lod = -1.0
        for epoch in range(200):
            
            if epoch < len(lods):
                l = lods[epoch]
            else:
                l = 0.0
                
            lod_G_assign_ops = tf.assign(G_lod_var, l)
            lod_E_assign_ops = tf.assign(E_lod_var, l)
            lod_D_img_assign_ops = tf.assign(D_img_lod_var, l)
            self.session.run([lod_G_assign_ops, lod_E_assign_ops, lod_D_img_assign_ops])

            if np.floor(l) != np.floor(prev_lod) or np.ceil(l) != np.ceil(prev_lod):
                reset_optimizer_op = tf.variables_initializer(optimizer_vars)
                self.session.run(reset_optimizer_op)
            prev_lod = l
            
            resolution = 2 ** (int(np.log2(self.size_image)) - int(np.floor(l)))
            batch_size = batch_sizes.get(resolution)
            num_batches = len(file_names) // batch_size
            
            for ind_batch in range(num_batches):
                
                batch_files = file_names[ind_batch*batch_size:(ind_batch+1)*batch_size]
                batch = [load_image(
                    image_path=batch_file,
                    lod=l,
                    image_size=resolution,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_files]
                if self.num_input_channels == 1:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                
                batch_files_mix = np.random.choice(file_names, batch_size)
                batch = [load_image(
                    image_path=batch_file,
                    lod=l,
                    image_size=resolution,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_files_mix]
                if self.num_input_channels == 1:
                    batch_images_mix = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images_mix = np.array(batch).astype(np.float32)
                
                _, _, rec_err_mix_c, rec_err_mix_m, rec_err, gan_d = self.session.run(
                    fetches = [
                        self.EG_optimizer,
                        self.D_optimizer,
                        self.EG_loss_mix_c,
                        self.EG_loss_mix_m,
                        self.EG_loss,
                        self.loss_D,
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.input_image_mix: batch_images_mix,
                    }
                )
                
                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]" %
                    (epoch, num_epochs, ind_batch, num_batches))
                print("lod=%.2f\tbatch_size=%d\tresolution=%d"%(l, batch_size, resolution))
                print("\n\tEG_err=%.4f"%(rec_err))
                print("\n\tEG_err_mix_coarse=%.4f"%(rec_err_mix_c))
                print("\n\tEG_err_mix_middle=%.4f"%(rec_err_mix_m))
                print("\n\tD_err=%.4f"%(gan_d))

            summary = self.summary.eval(
                feed_dict={
                    self.input_image: batch_images,
                    self.input_image_mix: batch_images_mix,
                }
            )

            self.writer.add_summary(summary, self.EG_global_step.eval())
            
            if np.mod(epoch, 5) == 0:    
                self.save_checkpoint()
                # save sample images for each epoch
                name = '{:02d}.png'.format(epoch)
                sample = [load_image(
                    image_path=sample_file,
                    lod=l,
                    image_size=resolution,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for sample_file in sample_files]
                if self.num_input_channels == 1:
                    sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
                else:
                    sample_images = np.array(sample).astype(np.float32)
                self.sample(sample_images[:sample_size], sample_images[sample_size:], name)

        self.save_checkpoint()
        self.writer.close()
        
    # Encoder: image --> z  ---------------------------------------------------
    def encoder(
        self,
        image,                          # First input: Images [minibatch, channel, height, width].
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        edlatent_broadcast  = None,
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
        use_wscale          = True,         # Enable equalized learning rate?
        dtype               = 'float32',    # Data type to use for activations and outputs.
        fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
        structure           = 'recursive',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
        is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
        reuse_variables=tf.AUTO_REUSE
        ):
      
        resolution = self.size_image
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        def blur(x): return ops.blur2d(x, blur_filter) if blur_filter else x
        act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[nonlinearity]
    
        image.set_shape([None, self.num_input_channels, resolution, resolution])
        image = tf.cast(image, dtype)
        
        with tf.variable_scope('E_lod', reuse=reuse_variables):
            E_lod_in = tf.cast(tf.get_variable('E_lod', initializer=np.float32(0), trainable=False), dtype)
    
        # Building blocks.
        def fromrgb(x, res): # res = 2..resolution_log2
            with tf.variable_scope('E_FromRGB_lod%d' % (resolution_log2 - res), reuse=reuse_variables):
                return act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
            
        def block(x, res): # res = 2..resolution_log2
            with tf.variable_scope('E_%dx%d' % (2**res, 2**res), reuse=reuse_variables):
                if res >= 3: # 8x8 and up
                    with tf.variable_scope('Conv0'):
                        x = act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Conv1_down'):
                        x = act(ops.apply_bias(ops.conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
                    return x
                
                else: # 4x4
                    with tf.variable_scope('Conv'):
                        x = act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Dense0'):
                        x = act(ops.apply_bias(ops.dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Dense1'):
                        latents = ops.apply_bias(ops.dense(x, fmaps=self.num_z_channels, gain=1, use_wscale=use_wscale))
                        #latents = tf.reshape(latents, [-1, 14, self.num_z_channels])
                    return tf.nn.tanh(latents)
    
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if structure == 'fixed':
            x = fromrgb(image, resolution_log2)
            for res in range(resolution_log2, 2, -1):
                x = block(x, res)
            latents = block(x, 2)
            
        # Linear structure: simple but inefficient.
        if structure == 'linear':
            img = image
            x = fromrgb(img, resolution_log2)
            for res in range(resolution_log2, 2, -1):
                lod = resolution_log2 - res
                x = block(x, res)
                img = ops.downscale2d(img)
                y = fromrgb(img, res - 1)
                with tf.variable_scope('E_Grow_lod%d' % lod):
                    x = ops.lerp_clip(x, y, E_lod_in - lod)
            latents = block(x, 2)
    
        # Recursive structure: complex but efficient.
        if structure == 'recursive':
            def cset(cur_lambda, new_cond, new_lambda):
                return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
            def grow(res, lod):
                x = lambda: fromrgb(ops.downscale2d(image, 2**lod), res)
                if lod > 0: x = cset(x, (E_lod_in < lod), lambda: grow(res + 1, lod - 1))
                x = block(x(), res); y = lambda: x
                if res > 2: y = cset(y, (E_lod_in > lod), lambda: ops.lerp(x, fromrgb(ops.downscale2d(image, 2**(lod+1)), res - 1), E_lod_in - lod))
                return y()
            latents = grow(2, resolution_log2 - 2)
        
        
        # Broadcast.
        if edlatent_broadcast is not None:
            with tf.variable_scope('E_Broadcast', reuse=reuse_variables):
                latents = tf.tile(latents[:, np.newaxis], [1, edlatent_broadcast, 1]) 
        

        assert latents.dtype == tf.as_dtype(dtype)
        latents = tf.identity(latents, name='latents')
    
        return latents

    # Generator/Decoder: w --> image --------------------------------------------------
    def decoder(
        self,
        w,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        use_styles          = True,         # Enable style inputs?
        const_input_layer   = True,         # First layer is a learned constant?
        use_noise           = True,         # Enable noise inputs?
        randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
        use_wscale          = True,         # Enable equalized learning rate?
        use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
        use_instance_norm   = True,         # Enable instance normalization?
        dtype               = 'float32',    # Data type to use for activations and outputs.
        fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
        structure           = 'recursive',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
        reuse_variables=tf.AUTO_REUSE):                         # Ignore unrecognized keyword args.
    
        resolution = self.size_image
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        def blur(x): return ops.blur2d(x, blur_filter) if blur_filter else x

        act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        images_out = None
    
        # Primary inputs.
        w.set_shape([None, num_styles, self.num_z_channels])
        w = tf.cast(w, dtype)
    
        with tf.variable_scope('G_lod', reuse=reuse_variables):
            G_lod_in = tf.cast(tf.get_variable('G_lod', initializer=np.float32(0), trainable=False), dtype)
    
        # Noise inputs.
        noise_inputs = []
        if use_noise:
            for layer_idx in range(num_layers):
                with tf.variable_scope('G_noise%d' % layer_idx, reuse=reuse_variables):
                    res = layer_idx // 2 + 2
                    shape = [1, use_noise, 2**res, 2**res]
                    noise_inputs.append(tf.get_variable('n%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))
    
        # Things to do at the end of each layer.
        def layer_epilogue(x, layer_idx):
            if use_noise:
                x = ops.apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise)
            x = ops.apply_bias(x)
            x = act(x)
            if use_pixel_norm:
                x = ops.pixel_norm(x)
            if use_instance_norm:
                x = ops.instance_norm(x)
            if use_styles:
                x = ops.style_mod(x, w[:, layer_idx], use_wscale=use_wscale)
            return x
    
        # Early layers.
        with tf.variable_scope('G_4x4', reuse=reuse_variables):
            if const_input_layer:
                with tf.variable_scope('Const'):
                    x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.ones())
                    x = layer_epilogue(tf.tile(tf.cast(x, dtype), [tf.shape(w)[0], 1, 1, 1]), 0)
            else:
                with tf.variable_scope('Dense'):
                    x = ops.dense(w[:, 0], fmaps=nf(1)*16, gain=gain/4, use_wscale=use_wscale) # tweak gain to match the official implementation of Progressing GAN
                    x = layer_epilogue(tf.reshape(x, [-1, nf(1), 4, 4]), 0)
            with tf.variable_scope('Conv'):
                x = layer_epilogue(ops.conv2d(x, fmaps=nf(1), kernel=3, gain=gain, use_wscale=use_wscale), 1)
    
        # Building blocks for remaining layers.
        def block(res, x): # res = 3..resolution_log2
            with tf.variable_scope('G_%dx%d' % (2**res, 2**res), reuse=reuse_variables):
                with tf.variable_scope('Conv0_up'):
                    x = layer_epilogue(blur(ops.upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)), res*2-4)
                with tf.variable_scope('Conv1'):
                    x = layer_epilogue(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale), res*2-3)
                return x
        def torgb(res, x): # res = 2..resolution_log2
            lod = resolution_log2 - res
            with tf.variable_scope('G_ToRGB_lod%d' % lod, reuse=reuse_variables):
                return ops.apply_bias(ops.conv2d(x, fmaps=self.num_input_channels, kernel=1, gain=1, use_wscale=use_wscale))
    
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if structure == 'fixed':
            for res in range(3, resolution_log2 + 1):
                x = block(res, x)
            images_out = torgb(resolution_log2, x)

        # Linear structure: simple but inefficient.
        if structure == 'linear':
            images_out = torgb(2, x)
            for res in range(3, resolution_log2 + 1):
                lod = resolution_log2 - res
                x = block(res, x)
                img = torgb(res, x)
                images_out = ops.upscale2d(images_out)
                with tf.variable_scope('G_Grow_lod%d' % lod):
                    images_out = ops.lerp_clip(img, images_out, G_lod_in - lod)
    
        # Recursive structure: complex but efficient.
        if structure == 'recursive':
            def cset(cur_lambda, new_cond, new_lambda):
                return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
            def grow(x, res, lod):
                y = block(res, x)
                img = lambda: ops.upscale2d(torgb(res, y), 2**lod)
                img = cset(img, (G_lod_in > lod), lambda: ops.upscale2d(ops.lerp(torgb(res, y), ops.upscale2d(torgb(res - 1, x)), G_lod_in - lod), 2**lod))
                if lod > 0: img = cset(img, (G_lod_in < lod), lambda: grow(y, res + 1, lod - 1))
                return img()
            images_out = grow(x, 3, resolution_log2 - 3)

        assert images_out.dtype == tf.as_dtype(dtype)
        return tf.identity(images_out, name='images_out')
    
    # Discriminator on Image: imahe --> score  --------------------------------
    def discriminator_img(
        self,
        image,                          # First input: Images [minibatch, channel, height, width].
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
        use_wscale          = True,         # Enable equalized learning rate?
        mbstd_group_size    = 3,            # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
        dtype               = 'float32',    # Data type to use for activations and outputs.
        fused_scale         = False,       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
        structure           = 'recursive',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
        reuse_variables=tf.AUTO_REUSE,
        ):
        
        resolution = self.size_image
        #mbstd_group_size = self.group_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        def blur(x): return ops.blur2d(x, blur_filter) if blur_filter else x
        act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (ops.leaky_relu, np.sqrt(2))}[nonlinearity]
    
        image.set_shape([None, self.num_input_channels, resolution, resolution])
        image = tf.cast(image, dtype)
        
        with tf.variable_scope('D_img_lod', reuse=reuse_variables):
            D_img_lod_in = tf.cast(tf.get_variable('D_img_lod', initializer=np.float32(0), trainable=False), dtype)
        
        # Building blocks.
        def fromrgb(x, res): # res = 2..resolution_log2
            with tf.variable_scope('D_img_FromRGB_lod%d' % (resolution_log2 - res), reuse=reuse_variables):
                return act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
            
        def block(x, res): # res = 2..resolution_log2
            with tf.variable_scope('D_img_%dx%d' % (2**res, 2**res), reuse=reuse_variables):
                if res >= 3: # 8x8 and up
                    with tf.variable_scope('Conv0'):
                        x = act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Conv1_down'):
                        x = act(ops.apply_bias(ops.conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
                    return x
                else: # 4x4
                    #out = x
                    #if mbstd_group_size > 1:
                    #    x = ops.minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                    with tf.variable_scope('Conv'):
                        x = act(ops.apply_bias(ops.conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                    with tf.variable_scope('Dense0'):
                        x = ops.apply_bias(ops.dense(x, fmaps=1, gain=1, use_wscale=use_wscale))
                    return x
        
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if structure == 'fixed':
            x = fromrgb(image, resolution_log2)
            for res in range(resolution_log2, 2, -1):
                x = block(x, res)
            logits_out = block(x, 2)
            
        # Linear structure: simple but inefficient.
        if structure == 'linear':
            img = image
            x = fromrgb(img, resolution_log2)
            for res in range(resolution_log2, 2, -1):
                lod = resolution_log2 - res
                x = block(x, res)
                img = ops.downscale2d(img)
                y = fromrgb(img, res - 1)
                with tf.variable_scope('D_img_Grow_lod%d' % lod):
                    x = ops.lerp_clip(x, y, D_img_lod_in - lod)
            logits_out = block(x, 2)

        # Recursive structure: complex but efficient.
        if structure == 'recursive':
            def cset(cur_lambda, new_cond, new_lambda):
                return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
            def grow(res, lod):
                x = lambda: fromrgb(ops.downscale2d(image, 2**lod), res)
                if lod > 0: x = cset(x, (D_img_lod_in < lod), lambda: grow(res + 1, lod - 1))
                x = block(x(), res); y = lambda: x
                if res > 2: y = cset(y, (D_img_lod_in > lod), lambda: ops.lerp(x, fromrgb(ops.downscale2d(image, 2**(lod+1)), res - 1), D_img_lod_in - lod))
                return y()
            logits_out = grow(2, resolution_log2 - 2)
    
        assert logits_out.dtype == tf.as_dtype(dtype)
        logits_out = tf.identity(logits_out, name='logits_out')

        return logits_out

    # maintanance -------------------------------------------------------------
    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )
        
    #--------------------------------------------------------------------------
    def load_checkpoint(self, model_path=None):
        if model_path is None:
            print("\n\tLoading pre-trained model ...")
            checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        else:
            print("\n\tLoading init model ...")
            checkpoint_dir = model_path
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            try:
                self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
                return True
            except:
                return False
        else:
            return False

    #--------------------------------------------------------------------------
    def sample(self, images, images_mix, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        G, G_xx_c, G_xx_m = self.session.run(
            [self.G, self.G_xx_c, self.G_xx_m],
            feed_dict={
                self.input_image: images,
                self.input_image_mix: images_mix
            }
        )
        
        
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[3,4]
        )
        
        
        save_batch_images(
            batch_images=G_xx_c,
            save_path=os.path.join(sample_dir, 'xx_c'+name),
            image_value_range=self.image_value_range,
            size_frame=[3,4]
        )
        
        save_batch_images(
            batch_images=G_xx_m,
            save_path=os.path.join(sample_dir, 'xx_m'+name),
            image_value_range=self.image_value_range,
            size_frame=[3,4]
        )        
        
    #--------------------------------------------------------------------------        
    def compute_fid(self):
        
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        file_names = glob(os.path.join('./', self.dataset_name, '*.jpg'))

        input_tensor = tf.keras.layers.Input(shape=(256, 256, 3))
        inception_model = tf.keras.applications.InceptionV3(input_tensor=input_tensor, include_top=False, pooling='avg')
        activations_real = np.empty([len(file_names), inception_model.output_shape[1]], dtype=np.float32)
        activations_fake = np.empty([len(file_names), inception_model.output_shape[1]], dtype=np.float32)
        
        batch_size = 8
        num_batches = len(file_names) // batch_size
        
        for ind_batch in range(num_batches):
            print(ind_batch, ' out of ', num_batches, '\n')
            begin = ind_batch * batch_size
            end = (ind_batch + 1) * batch_size
            batch_files = file_names[begin:end]
            
            batch = [load_image(
                image_path=batch_file,
                lod=0.0,
                image_size=256,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            ) for batch_file in batch_files]
            if self.num_input_channels == 1:
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)
                
            G = self.session.run(
                self.G,
                feed_dict={
                    self.input_image: batch_images
                }
            )
                
            batch_images = batch_images.transpose([0, 2, 3, 1]) # (n, 3, 256, 256) => (n, 256, 256, 3)
            G = G.transpose([0, 2, 3, 1]) # (n, 3, 256, 256) => (n, 256, 256, 3)
            activations_real[begin:end, :] = inception_model.predict(batch_images)
            activations_fake[begin:end, :] = inception_model.predict(G)
        
        mu_real = np.mean(activations_real, axis=0)
        sigma_real = np.cov(activations_real, rowvar=False)
        mu_fake = np.mean(activations_fake, axis=0)
        sigma_fake = np.cov(activations_fake, rowvar=False)
        
        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        
        print("FID Score is ", np.real(dist))
        
    #--------------------------------------------------------------------------    
    def compute_ppl(self):
        
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        file_names = glob(os.path.join('./', self.dataset_name, '*.jpg'))
        batch_size = 8
        num_batches = len(file_names) // batch_size

        distance_measure = lpips.LPIPS(net='vgg', use_dropout=False)
        
        for ind_batch in range(num_batches):   
            begin = ind_batch * batch_size
            end = (ind_batch + 1) * batch_size
            batch_files = file_names[begin:end]
            
            batch = [load_image(
                image_path=batch_file,
                lod=0.0,
                image_size=256,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            ) for batch_file in batch_files]
            if self.num_input_channels == 1:
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)
                
            w_t01 = self.session.run(
                self.w_x,
                feed_dict={
                    self.input_image: batch_images
                }
            )
        
            # Generate random latents and interpolation t-values.
            sampling = 'full' # or sampling = 'end'
            lerp_t = np.random.uniform(0.0, 1.0 if sampling == 'full' else 0.0, [int(batch_size/2)])
            epsilon = 10e-4
            # Interpolate in W
            w_t0, w_t1 = w_t01[0::2], w_t01[1::2]
            w_e0 = ops.lerp(w_t0, w_t1, lerp_t[:, np.newaxis, np.newaxis])
            w_e1 = ops.lerp(w_t0, w_t1, lerp_t[:, np.newaxis, np.newaxis] + epsilon)
            w_e01 = np.reshape(np.stack((w_e0, w_e1), axis=1), w_t01.shape)
            # Synthesize images.
            G = self.session.run(
                self.G,
                feed_dict={
                    self.w_x: w_e01
                }
            )
        
            # Crop only the face region.
            c = int(G.shape[2])
            h = (7.0 - 3.0) / 8.0 * (2.0 / 1.6410)
            w = (6.0 - 2.0) / 8.0 * (2.0 / 1.6410)
            vc = (7.0 + 3.0) / 2.0 / 8.0
            hc = (6.0 + 2.0) / 2.0 / 8.0
            h = int(h * c)
            w = int(w * c)
            hc = int(hc * c)
            vc = int(vc * c)
            G = G[:, :, vc - h // 2: vc + h // 2, hc - w // 2: hc + w // 2]
            
            # Scale dynamic range from [-1,1] to [0,255] for VGG.
            G = (G + 1) * (255 / 2)
            # Evaluate perceptual distance.
            img_e0, img_e1 = G[0::2], G[1::2]  
            img_e0 = torch.tensor(img_e0)
            img_e1 = torch.tensor(img_e1)
            
            d = distance_measure(img_e0, img_e1) * (1 / epsilon**2)
            d = d.cpu().detach().numpy()
            d = np.reshape(d, (4,))
            if ind_batch == 0:
                all_distances = d
            else:
                all_distances = np.concatenate((all_distances, d), 0)
        
        # Reject outliers.
        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)
        
        print("PPL Score for w is ", np.mean(filtered_distances))
  #--------------------------------------------------------------------------
    
    