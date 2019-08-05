# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:37:05 2019

@author: Balaji
"""
from keras import layers
from keras.layers import Input, Dense,SeparableConv2D,BatchNormalization,Activation,Dropout,Conv2D,concatenate,MaxPooling2D,Lambda,AveragePooling2D,Flatten
import tensorflow as tf
from keras.models import Model

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

def space_to_depth_x4(x):
    return tf.space_to_depth(x, block_size=4)


def network(img_height,img_width,channel):    
    input_layer = Input(shape=(img_height, img_width, channel))
    
    # Layer 1, 5x5 Separable , Depthwise
    Layer1 = SeparableConv2D(32, (5,5), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_layer)
    Layer1_normalized = BatchNormalization(name='norm_1')(Layer1)
    Layer1_normalized_activated = Activation('relu')(Layer1_normalized)
    
    Layer1_normalized_activated = Dropout(0.2)(Layer1_normalized_activated)
    
    # Layer 2,Normal Conv2d 5x5
    Layer2 = Conv2D(64, (5,5), strides=(1,1), padding='same', name='conv_2', use_bias=False)(Layer1_normalized_activated)
    Layer2_normalized = BatchNormalization(name='norm_2')(Layer2)
    Layer2_normalized_activated = Activation('relu')(Layer2_normalized)
    
    Layer2_normalized_activated = Dropout(0.2)(Layer2_normalized_activated)
    
    # Layer 3, Normal Conv2d 5x5
    Layer3 = Conv2D(64, (5,5), strides=(1,1), padding='same', name='conv_3', use_bias=False)(Layer2_normalized_activated)
    Layer3_normalized = BatchNormalization(name='norm_3')(Layer3)
    Layer3_normalized_activated = Activation('relu')(Layer3_normalized)
    Layer3_normalized_activated = Dropout(0.2)(Layer3_normalized_activated)
    
    Layers_1_3_concatenated = concatenate([Layer1_normalized_activated,Layer3_normalized_activated])
    
    # Layer 4, 5x5 Separable
    Layer4 = SeparableConv2D(64, (5,5), strides=(1,1), padding='same', name='conv_4', use_bias=False)(Layers_1_3_concatenated)
    Layer4_normalized = BatchNormalization(name='norm_4')(Layer4)
    Layer4_normalized_activated = Activation('relu')(Layer4_normalized)
    Layer4_normalized_activated = Dropout(0.2)(Layer4_normalized_activated)
    
    Layers_1_4_concatenated = concatenate([Layer1_normalized_activated,Layer4_normalized_activated])
    
    #Maxpool layer
    Layer_maxpool1 = MaxPooling2D(pool_size=(2, 2))(Layers_1_4_concatenated) #16
    
    #Bottleneck 1x1 to reduce params
    Layer_bottleneck1 = Conv2D(32, (1, 1), activation='relu')(Layer_maxpool1)
    
    # Layer 5, 3x3 Separable , Depthwise
    Layer5 = SeparableConv2D(64, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(Layer_bottleneck1)
    Layer5_normalized = BatchNormalization(name='norm_5')(Layer5)
    Layer5_normalized_activated = Activation('relu')(Layer5_normalized)
    Layer5_normalized_activated = Dropout(0.2)(Layer5_normalized_activated)
    
    Layers_1_4_5_concatenated = concatenate([Lambda(space_to_depth_x2)(Layer1_normalized_activated),Lambda(space_to_depth_x2)(Layer4_normalized_activated),Layer5_normalized_activated])
    
    # Layer 6, Normal Conv2d 5x5
    Layer6 = Conv2D(64, (5,5), strides=(1,1), padding='same', name='conv_6', use_bias=False)(Layers_1_4_5_concatenated)
    Layer6_normalized = BatchNormalization(name='norm_6')(Layer6)
    Layer6_normalized_activated = Activation('relu')(Layer6_normalized)
    Layer6_normalized_activated = Dropout(0.2)(Layer6_normalized_activated)
    
    Layers_3_4_5_6_concatenated = concatenate([Lambda(space_to_depth_x2)(Layer3_normalized_activated),Lambda(space_to_depth_x2)(Layer4_normalized_activated),Layer5_normalized_activated,Layer6_normalized_activated])
    
    # Layer 7, 3x3 Separable , Depthwise
    Layer7 = SeparableConv2D(64, (3,3), strides=(1,1), padding='same', name='conv_7', use_bias=False)(Layers_3_4_5_6_concatenated)
    Layer7_normalized = BatchNormalization(name='norm_7')(Layer7)
    Layer7_normalized_activated = Activation('relu')(Layer7_normalized)
    Layer7_normalized_activated = Dropout(0.2)(Layer7_normalized_activated)
    
    Layers_1_3_4_5_6_7_concatenated = concatenate([Lambda(space_to_depth_x2)(Layer1_normalized_activated),Lambda(space_to_depth_x2)(Layer3_normalized_activated),Lambda(space_to_depth_x2)(Layer4_normalized_activated),Layer5_normalized_activated,Layer6_normalized_activated,Layer7_normalized_activated])
    
    # Layer 8, 5x5 Separable Depthwise.
    Layer8 = SeparableConv2D(64, (5,5), strides=(1,1), padding='same', name='conv_8', use_bias=False)(Layers_1_3_4_5_6_7_concatenated)
    Layer8_normalized = BatchNormalization(name='norm_8')(Layer8)
    Layer8_normalized_activated = Activation('relu')(Layer8_normalized)
    Layer8_normalized_activated = Dropout(0.2)(Layer8_normalized_activated)
    
    Layers_1_4_7_8_concatenated = concatenate([Lambda(space_to_depth_x2)(Layer1_normalized_activated),Lambda(space_to_depth_x2)(Layer4_normalized_activated),Layer7_normalized_activated,Layer8_normalized_activated])
    
    Layer_maxpool2 = MaxPooling2D(pool_size=(2, 2))(Layers_1_4_7_8_concatenated) #8
    
    Layers_6_maxpool2_concatenated = concatenate([Lambda(space_to_depth_x2)(Layer6_normalized_activated),Layer_maxpool2])
    
    #Bottleneck 1x1 to reduce params
    Layer_bottleneck2 = Conv2D(64, (1, 1), activation='relu')(Layers_6_maxpool2_concatenated)
    
    # Layer 9
    Layer9 = Conv2D(64, (5,5), strides=(1,1), padding='same', name='conv_9', use_bias=False)(Layer_bottleneck2)
    Layer9_normalized = BatchNormalization(name='norm_9')(Layer9)
    Layer9_normalized_activated = Activation('relu')(Layer9_normalized)
    Layer9_normalized_activated = Dropout(0.2)(Layer9_normalized_activated)
    
    Layers_1_2_7_9_concatenated = concatenate([Lambda(space_to_depth_x4)(Layer1_normalized_activated),Lambda(space_to_depth_x4)(Layer2_normalized_activated),Lambda(space_to_depth_x2)(Layer7_normalized_activated),Layer9_normalized_activated])
    
    # Layer 10,5x5 Separable , Depthwise
    Layer10 = SeparableConv2D(64, (5,5), strides=(1,1), padding='same', name='conv_10', use_bias=False)(Layers_1_2_7_9_concatenated)
    Layer10_normalized = BatchNormalization(name='norm_10')(Layer10)
    Layer10_normalized_activated = Activation('relu')(Layer10_normalized)
    Layer10_normalized_activated = Dropout(0.2)(Layer10_normalized_activated)
    
    Layers_2_3_5_9_10_concatenated = concatenate([Lambda(space_to_depth_x4)(Layer2_normalized_activated),Lambda(space_to_depth_x4)(Layer3_normalized_activated),Lambda(space_to_depth_x2)(Layer5_normalized_activated),Layer9_normalized_activated,Layer10_normalized_activated])
    
    # Layer 11, Normal Conv2d 3x3
    Layer11 = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(Layers_2_3_5_9_10_concatenated)
    Layer11_normalized = BatchNormalization(name='norm_11')(Layer11)
    Layer11_normalized_activated = Activation('relu')(Layer11_normalized)
    Layer11_normalized_activated = Dropout(0.2)(Layer11_normalized_activated)
    
    Layers_3_4_5_7_10_11_concatenated = concatenate([Lambda(space_to_depth_x4)(Layer3_normalized_activated),Lambda(space_to_depth_x4)(Layer4_normalized_activated),Lambda(space_to_depth_x2)(Layer5_normalized_activated),Lambda(space_to_depth_x2)(Layer7_normalized_activated),Layer10_normalized_activated,Layer11_normalized_activated])
    
    # Layer 12 -  5x5 Separable , Depthwise
    Layer12 = SeparableConv2D(64, (3,3), strides=(1,1), padding='same', name='conv_12', use_bias=False)(Layers_3_4_5_7_10_11_concatenated)
    Layer12_normalized = BatchNormalization(name='norm_12')(Layer12)
    Layer12_normalized_activated = Activation('relu')(Layer12_normalized)
    Layer12_normalized_activated = Dropout(0.2)(Layer12_normalized_activated)
    
    Layers_4_7_10_12_concatenated = concatenate([Lambda(space_to_depth_x4)(Layer4_normalized_activated), Lambda(space_to_depth_x2)(Layer7_normalized_activated), Layer10_normalized_activated, Layer12_normalized_activated])
    
    Layer_bottleneck3 = Conv2D(10, (1,1), strides=(1,1), padding='same', name='conv_f1', use_bias=False)(Layers_4_7_10_12_concatenated)
    #Layer_f2 = Conv2D(10, (8,8), strides=(1,1), name='conv_f2', use_bias=False)(Layer_bottleneck3)
    
    img_output = AveragePooling2D(8,8)(Layer_bottleneck3)
    img_output=  Flatten()(img_output)
    img_output = Activation('softmax')(img_output)
    
    model = Model(inputs=[input_layer], outputs=[img_output])
    model.summary()

network(32,32,3)