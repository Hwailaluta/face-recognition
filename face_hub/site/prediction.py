class Model:
    def __init__(self):
        import numpy as np
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
        from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Input, add
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.regularizers import l1, l2
        from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
        from tensorflow.keras.optimizers import SGD
        from tensorflow.keras.layers import Dropout


        self.columns = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        def res_block(layer, filters, stride=(2,2)):
            x = BatchNormalization()(layer)
            x = Activation('relu')(x)
            x = Conv2D(filters, stride, activation = 'relu')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filters, stride, activation = 'relu')(x)
            return x

        inputs = Input(shape=(64, 64, 3))
        saved_inputs = inputs

        # I've left out the initial batch normalization to keep variance in the input information.
        # I've also removed the regularizers.
        x = Conv2D(64, (1, 1), use_bias=False)(inputs)

        x = BatchNormalization(axis = 1, epsilon = 0.0001, momentum = 0.95)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x) # This layer is a bottleneck.
        # Same padding prevents downsampling so we can add the original input back in later.

        x = BatchNormalization(axis = 1, epsilon = 0.0001, momentum = 0.95)(x)
        x = Activation("relu")(x)
        x = Conv2D(128, (1, 1), use_bias=False)(x)

        saved_inputs = Conv2D(128, (1, 1), use_bias=False)(saved_inputs)
        x = add([x, saved_inputs])

        y = res_block(x, 64)
        x = Conv2D(64, (2, 2), activation = 'relu')(x)
        x = Conv2D(64, (2, 2), activation = 'relu')(x)
        x = add([x, y])

        y = res_block(x, 64)
        x = Conv2D(64, (2, 2), activation = 'relu')(x)
        x = Conv2D(64, (2, 2), activation = 'relu')(x)
        x = add([x, y])

        y = res_block(x, 128, (1, 1))
        x = Conv2D(128, (1, 1), activation = 'relu')(x)
        x = Conv2D(128, (1, 1), activation = 'relu')(x)
        x = add([x, y])

        y = res_block(x, 128, (1, 1))
        x = Conv2D(128, (1, 1), activation = 'relu')(x)
        x = Conv2D(128, (1, 1), activation = 'relu')(x)
        x = add([x, y])

        x = Dropout(0.2)(x)
        x = ZeroPadding2D((1,1))(x)
        x = MaxPooling2D((2,2), strides=(2,2))(x)
        # I've added dropout and 'blurred' the image with max pooling before the last half of the network.

        y = res_block(x, 256, (2, 2))
        x = Conv2D(256, (2, 2), activation = 'relu')(x)
        x = Conv2D(256, (2, 2), activation = 'relu')(x)
        x = add([x, y])

        y = res_block(x, 256, (2, 2))
        x = Conv2D(256, (2, 2), activation = 'relu')(x)
        x = Conv2D(256, (2, 2), activation = 'relu')(x)
        x = add([x, y])

        y = res_block(x, 512, (2, 2))
        x = Conv2D(512, (2, 2), activation = 'relu')(x)
        x = Conv2D(512, (2, 2), activation = 'relu')(x)
        x = add([x, y])

        y = res_block(x, 512, (2, 2))
        x = Conv2D(512, (2, 2), activation = 'relu')(x)
        x = Conv2D(512, (2, 2), activation = 'relu')(x)
        x = add([x, y])

        x = BatchNormalization(axis = 1, epsilon = 0.0001, momentum = 0.95)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(x)
        x = Flatten()(x)

        output = Dense(40, activation = 'sigmoid', kernel_initializer='he_normal')(x)

        self.model = Model(inputs, output, name="ResNet_for_faces")
        sgd = SGD(momentum=0.9, nesterov=True)
        self.model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

        self.model.load_weights('res_512_weights.h5')


# prediction = model.predict(image)
# print(prediction)

# face_features = {}
# for x, pred in enumerate(prediction):
#     face_features[x] = []
#     for i, prob in enumerate(pred):
#         if prob >= 0.5:
#             face_features[x].append(columns[i])

# print(face_features)