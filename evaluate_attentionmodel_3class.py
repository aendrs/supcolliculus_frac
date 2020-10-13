# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:08:15 2019

@author: andre
"""

import os
os.chdir('C:\\Users\\andre\\Documents\\Torino\\colliculus')

from keras.models import  Model
from keras.layers import (
        Input, Lambda, Dense, Conv2D, BatchNormalization, Activation, 
        concatenate, MaxPooling2D, AveragePooling2D, Flatten, Dropout, GlobalAveragePooling2D, SpatialDropout2D)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad, Adam
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

from fractalnet import fractal_net
import attention_mechanism
import vis_funcs

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
import numpy as np
import time
import sys


#%%  =========================================================================
# Load the model using the unreliable method (in lieu of the proper one)
# due to the incapacity of keras to save/load our model we have to first 
# reconstruct the model from the original script. So go to the relevant script 

model=[] #this is just to avoid the stupid warnings in Spyder IDE

script='Colliculus_FractalAttention_3classes_03.py' #'Colliculus_FractalAttention_20.py'
pathtomodel='.\\trained_models\\2020-10-13' #'.\\2019-04-17'
#
K.set_learning_phase(False)


# This is a hacky way to load the model structure from another script, (due to the fact that we have package problems when loading)
# pay attention to the beginning and end lines!!!!
begline=62 #54
endline=239 #233 217
txt = ''
with open(script) as sc:
    for i, line in enumerate(sc):
        if i >= begline and i<= endline:
            txt = txt + line

exec(txt)


from keras.utils import plot_model
plot_model(model, to_file=pathtomodel + '\\' + 'model.png', show_layer_names=True, show_shapes=True)


# Load Weights
model.load_weights(pathtomodel+"\\model.h5") #model.h5 are the WEIGHTS




#%% ==========================================================================

# for dataset_V6
imagesdict={'happy':['0001','0006','0050','0059','0118','0119','0168','0174','0218','0190','0188'],
        'neutral':['0003','0008','0012','0020','0046','0051','0058','0082','0087','0143','0223','0248'],
        'sad':['0002','0003','0021','0024','0048','0066','0070','0103','0141','0174','0212','0218','0239']}


dataset= 'datasets\\dataset_V6_PMK_gauss2_3class' #'v6_PMK_gauss2_augmented'
basepath= 'C:\\Users\\andre\\Documents\\Torino\\colliculus' #'E:\\andres\\datasets'
train_or_test='test'
                   
dataset_baseimage='datasets\\dataset_V6'    # 'dataset_V6, dataset with base image not modified by Diff of Gaussians, for visualization purposes

labels_dict={'happy':0, 'neutral':1, 'sad':2}

#%%

savedirectory='AttentionImages'

start=time.time()
for category, imagelist in imagesdict.items():
    print('category: {}'.format(category))
    for image_number in imagelist:
        print('working on image: {}  {}'.format(category, image_number))
        
        #check if savedirectory exists otherwise create it
        finalsavedirectory=pathtomodel+'\\'+savedirectory+'\\'+category
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
        
# load base image============================================================== 
        baseimage = skimage.io.imread(basepath + '\\' + dataset_baseimage + '\\' + train_or_test + '\\' +  category + '\\' + train_or_test + '_' + category + '_' + image_number + '.jpg')
        baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
        #vis_funcs.display_images([baseimage], cols=2)

        
        # load PMK streams images as a list of tensors
        # imagePMK list of three tensors shape:128,128,3
        imagePMK=vis_funcs.load_imagePMK(image_number, category, 
                           image_size=(128,128),
                           dataset=dataset, 
                           basepath=basepath, 
                           train_or_test=train_or_test )

# visualize Base Image next to the PMK streams=================================
        imagesAll=[baseimage]
        imagesAll.extend(imagePMK)
        vis_funcs.display_images(imagesAll, cols=4, titles=['Base Image', 'P', 'M', 'K'])
        plt.savefig(finalsavedirectory+'\\{}_{}_base.png'.format(category,image_number),dpi=100)
                
# Preprocess imagePMK and get the predictions for the imagePMK ================
    #convert images to float32
        imagePMK=[image.astype(np.float32) for image in imagePMK]
    # Make it a batch of one. The model expects a batch, not a single image
        imagePMK=[image[np.newaxis,...] for image in imagePMK]
    #remember tha the input tensor needs to be [0,1]
        imagePMK=[image/255. for image in imagePMK]

        predictions =vis_funcs.predict3emotions(model, imagePMK) #!!!!!!!!!!!!!!!!!!!!! ATTENTION, 3 images insted of 5
        
        label_index = np.argmax(predictions) #useful for later, the label of the max category
    
    
    
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10, 5))
        ax0.axis("off")
        ax0.set_title('Image: {} {}'.format(category,image_number), fontsize=14)
        ax0.imshow(vis_funcs.normalize(baseimage), interpolation=None)
        #=====
        ax1.axis("off")
        ax1.imshow(vis_funcs.normalize(baseimage)*0.1+0.85)
        #plt.colorbar(im1,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
        #ax1.set_title('YYYYY', fontsize=10)
        ax1.text(30, 65, 
            '''predictions: \n 
              happy {:.3f} \n
              neutral {:.3f} \n
              sad {:.3f}'''.format(predictions[0][0],predictions[0][1], 
              predictions[0][2]),
              ha="left", va="center",color='k' , fontsize=14)
        plt.savefig(finalsavedirectory+'\\{}_{}_predictions.png'.format(category,image_number),dpi=100)
        
        
#  Heatmap visualization ======================================================
#  based on an sliding occlusion of the image and step-wise classification at each occlusion position
        rowscols=10
        imagePMK_hm=[vis_funcs.occlusion_tensor(image,rowscols=rowscols,squarecolor=[0.,0.,0.]) for image in imagePMK]
        heatmap_y = model.predict(imagePMK_hm)

        # Histogram of the classification probabilities across occlusion cases (for the right class)
        probs = heatmap_y[:, label_index] #choose the label index obtained with the classification of the complete image
        vis_funcs.tensor_summary(probs) 
        fig=plt.figure(figsize=(8,5)) 
        plt.hist(probs)
        plt.title('Histogram of classification probabilities \n Image: {} {}'.format(category,image_number))
        #plt.show()
        plt.savefig(finalsavedirectory+'\\{}_{}_occlusion_histogram.png'.format(category,image_number),dpi=100)
            
        # normalize probabilities [0,1] to construct the mask
        mask=[]
        mask = (probs.max() - probs) / (probs.max()-probs.min())
        mask = np.reshape(mask, (rowscols, rowscols)) #reshape probs vector
        vis_funcs.tensor_summary(mask)    
        
        #vis_funcs.tensor_summary(mask)
        #_ = plt.imshow(mask, cmap=plt.cm.Reds)    
    

        probs_matrix = np.reshape(probs, (rowscols, rowscols)) #reshape the probs vector (original values, no normalization like before)
           
        vis_funcs.apply_mask_heatmap(baseimage, mask, probs_matrix, order=1, cmap='OrRd_r',
                                     title1='Image: {} {}'.format(category,image_number))   
        plt.suptitle('Occlusion technique: classification by occluding single facial features'.format(category,image_number))
        plt.savefig(finalsavedirectory+'\\{}_{}_occlusion_heatmap_probs.png'.format(category,image_number),dpi=100)
    
    

    
# variation from baseline when occluded (pyramidal scale ) ====================================      

        # interpolation order=1
        vis_funcs.occlusion_variationfrombaseline(model,imagePMK,baseimage, label_index, 
                                                  predictions,category,image_number,
                                                  rowscols_list=[5,7,9,10],order=1)
        plt.savefig(finalsavedirectory+'\\{}_{}_occlusion_VARIATION.png'.format(category,image_number),dpi=100)
        
        
        # interpolation order=0
        vis_funcs.occlusion_variationfrombaseline(model,imagePMK,baseimage, label_index, 
                                                  predictions,category,image_number,
                                                  rowscols_list=[5,7,9,10],order=0)
        plt.savefig(finalsavedirectory+'\\{}_{}_occlusion_VARIATION_order0.png'.format(category,image_number),dpi=100)
        
        
# bubbles visualization, i.e., just showing specific features, instead of occluding them (inspired by Schynz 2001?)   
        rowscols=4
        imagePMK_bubbles=[vis_funcs.bubbles_mask_tensor(image,rowscols=rowscols,gaussian=True, sigma=18, biasvalue=0.5) for image in imagePMK]
        bubbles_y = model.predict(imagePMK_bubbles)       
        
        
        probs = bubbles_y[:, labels_dict[category]] #choose the CORRECT label index, not the max prob, based on the category name of the image
        vis_funcs.tensor_summary(probs) 
        
        '''
        fig=plt.figure(figsize=(8,5)) 
        plt.hist(probs)
        plt.title('Bubbles: Histogram of classification probabilities \n Image: {} {}'.format(category,image_number))
        '''
            
        # normalize probabilities [0,1] to construct the mask
        mask=[]
        mask = vis_funcs.normalize(probs)
        mask = np.reshape(mask, (rowscols, rowscols)) #reshape probs vector
        vis_funcs.tensor_summary(mask)    

        probs_matrix = np.reshape(probs, (rowscols, rowscols)) #reshape the probs vector (original values, no normalization like before)
           
        vis_funcs.apply_mask_heatmap(baseimage, mask, probs_matrix, order=0, cmap='OrRd',
                                     title1='Image: {} {}'.format(category,image_number),
                                     title2= 'Classification probability when exclusively presented')   
        plt.suptitle('Bubbles technique: classification by showing single facial features'.format(category,image_number))
        plt.savefig(finalsavedirectory+'\\{}_{}_bubbles.png'.format(category,image_number),dpi=100)
        
      
        #plt.imshow(imagePMK_bubbles[0][np.argsort(probs)[-1],...]) #show the bubble with the highest classification probability

        
        vis_funcs.display_images(imagePMK_bubbles[0], cols=rowscols)
        plt.suptitle('Bubbles techinque for classification \n Image: {} {}'.format(category,image_number))
        plt.savefig(finalsavedirectory+'\\{}_{}_bubbles_contactsheet.png'.format(category,image_number),dpi=100)
    
# bubbles pyramidal analysis =================================================
        
        correct_category_label=labels_dict[category]
        vis_funcs.bubbles_pyramid(model,imagePMK,baseimage, correct_category_label, predictions,
                    category, image_number, rowscols_list=[5], sigma_list=[18,18,18,18], biasvalue=0.4, order=1, cmap='jet')
        plt.savefig(finalsavedirectory+'\\{}_{}_bubbles_pyramid.png'.format(category,image_number),dpi=100)
    
    
#==============================================================================    
# Attention Layer Activations =================================================   
        # get all attention masks and combine them in a single one    
        layer_names=['reshape_2','reshape_5','reshape_8','reshape_11'] #,'reshape_14' #!!!!!!!!! (always check the model's graph plot in order to be sure which)
        masks=[]
        for layer in layer_names:
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            mask=np.sum(outputs, axis=-1) #sum all the masks from a single AttModule to get a single 2D mask
            mask=mask.squeeze()
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            masks.append(mask)
        masks=np.stack(masks) # stack the resulting masks from the AttModules
                 
        mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
        mask=vis_funcs.normalize(mask)
         
        '''
        vis_funcs.apply_mask(baseimage, mask**2, avoid_black=True,  imagemultiplier=0.1, 
                       include_side_image=True, cmap="jet", title='Attention Salience')
        plt.savefig(finalsavedirectory+'\\{}_{}_Attention_sq.png'.format(category,image_number),dpi=100)
    
        vis_funcs.apply_mask(baseimage, mask, avoid_black=True,  imagemultiplier=0.1, 
                       include_side_image=True, cmap="jet", title='Attention Salience') 
        plt.savefig(finalsavedirectory+'\\{}_{}_Attention.png'.format(category,image_number),dpi=100)
        '''
        
        #overlay of mask on image with side colorbar
        fig=plt.figure(figsize=(6,4))    
        plt.axis('off')
        im1=plt.imshow(vis_funcs.normalize(baseimage))   
        im2=plt.imshow(mask,alpha=0.7,cmap='jet')   
        plt.title('Salience Overlay V1\n Image: {} {}'.format(category,image_number),fontsize=12)
        plt.colorbar(im2,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
        plt.savefig(finalsavedirectory+'\\{}_{}_Attention_overlay_V1.png'.format(category,image_number),dpi=100)
        plt.show()



        
        titles=['uSGS_01', 'lSGS_01','lSGS_02','SO'] #'uSGS_02',
        fig, axes = plt.subplots(2,3, figsize=(15, 10))
        # overall composite mask over image
        axes[0,0].axis("off")
        axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
        axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
        axes[0,0].set_title('Total Salience map V1', fontsize=14, color='r')  
        for n,layer in enumerate(layer_names):
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            outputs[outputs<.7]=0 #to avoid problems when doing boolean operations
            mask=np.sum(outputs, axis=-1) #sum all the masks from a single AttModule to get a single 2D mask  
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            i=0 if n<2 else 1
            j=n+1 if n<2 else n-2
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
            axes[i,j].axis("off")
            #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
            axes[i,j].set_title(titles[n], fontsize=14)               
            axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
        plt.suptitle('Salience Maps V1 \n Image: {} {}'.format(category,image_number))    
        plt.savefig(finalsavedirectory+'\\{}_{}_SalienceMaps_by_SClayer.png'.format(category,image_number),dpi=100)
        plt.show()     





#======= SECOND WAY TO COMBINE MASKS!!!!!
        # get all attention masks and combine them in a single one    
        layer_names=['reshape_2','reshape_5','reshape_8','reshape_11']  #,'reshape_14' 
        masks=[]
        for layer in layer_names:
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            outputs[outputs<.7]=0 #threshold, useful for the first layers, pay attention
            mask=np.any(outputs, axis=-1) #Logical OR across last dimension
            mask=mask.squeeze()
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            masks.append(mask)
        masks=np.stack(masks) # stack the resulting masks from the AttModules
        mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
        mask=vis_funcs.normalize(mask) #normaize to [0,1]


        #overlay of mask on image with side colorbar
        fig=plt.figure(figsize=(6,4))    
        plt.axis('off')
        im1=plt.imshow(vis_funcs.normalize(baseimage))   
        im2=plt.imshow(mask,alpha=0.7,cmap='jet')   
        plt.title('Salience Overlay V2 \n Image: {} {}'.format(category,image_number),fontsize=12)
        plt.colorbar(im2,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
        plt.savefig(finalsavedirectory+'\\{}_{}_Attention_overlay_V2.png'.format(category,image_number),dpi=100)
        plt.show()


        titles=['uSGS_01', 'lSGS_01','lSGS_02','SO']# 'uSGS_02',
        fig, axes = plt.subplots(2,3, figsize=(15, 10))
        # overall composite mask over image (previously calculated)
        axes[0,0].axis("off")
        axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
        axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
        axes[0,0].set_title('Total Salience map V2', fontsize=14, color='r')  
        #calcuate the rest
        for n,layer in enumerate(layer_names):
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            outputs[outputs<.7]=0 #to avoid problems when doing boolean operations
            mask=np.any(outputs, axis=-1)
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            i=0 if n<2 else 1
            j=n+1 if n<2 else n-2
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
            axes[i,j].axis("off")
            #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
            axes[i,j].set_title(titles[n], fontsize=14)               
            axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
        plt.suptitle('Salience Maps V2 \n Image: {} {}'.format(category,image_number))    
        plt.savefig(finalsavedirectory+'\\{}_{}_SalienceMaps_by_SClayer_2ndProcessing.png'.format(category,image_number),dpi=100)
        plt.show()   


end=time.time()-start 
print('... \n ===== {:.2f} minutes ===='.format(end/60))     



      
#%% ==========================================================================
#  ==========================================================================
#  Complete and correct BUBBLES implementation, just as in the paper by Gosselin and Schyns 2001


# Clean and streamline all this shiiiiit !!!!!!!!!!!!!!!!!!!!!!!!1


savedirectory='Bubbles_50'

start=time.time()
for category, imagelist in sorted(imagesdict.items()):
    print('category: {}'.format(category))
    for image_number in imagelist:
        print('working on image: {}  {}'.format(category, image_number))
        #check if savedirectory exists otherwise create it
        finalsavedirectory=pathtomodel+'\\'+savedirectory+'\\'+category
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
# load base image============================================================== 
        baseimage = skimage.io.imread(basepath + '\\' + dataset_baseimage + '\\' + train_or_test + '\\' +  category + '\\' + train_or_test + '_' + category + '_' + image_number + '.jpg')
        baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
        #vis_funcs.display_images([baseimage], cols=2)
        # load PMK streams images as a list of tensors
        # imagePMK list of three tensors shape:128,128,3
        imagePMK=vis_funcs.load_imagePMK(image_number, category, 
                           image_size=(128,128),
                           dataset=dataset, 
                           basepath=basepath, 
                           train_or_test=train_or_test )
# visualize Base Image next to the PMK streams=================================
        imagesAll=[baseimage]
        imagesAll.extend(imagePMK)
        vis_funcs.display_images(imagesAll, cols=4, titles=['Base Image', 'P', 'M', 'K'])
# Preprocess imagePMK and get the predictions for the imagePMK ================
    #convert images to float32
        imagePMK=[image.astype(np.float32) for image in imagePMK]
    # Make it a batch of one. The model expects a batch, not a single image
        imagePMK=[image[np.newaxis,...] for image in imagePMK]
    #remember that the input tensor needs to be [0,1]
        imagePMK=[image/255. for image in imagePMK]
        
        predictions =vis_funcs.predict3emotions(model, imagePMK) # !!!!!!!!!!!!!!!!!!!!!!!
        
        label_index = np.argmax(predictions) #useful for later, the label of the max category
        
        
        correct_label=labels_dict.get(category)
    
    
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10, 5))
        ax0.axis("off")
        ax0.set_title('Image: {} {}'.format(category,image_number), fontsize=14)
        ax0.imshow(vis_funcs.normalize(baseimage), interpolation=None)
        #=====
        ax1.axis("off")
        ax1.imshow(vis_funcs.normalize(baseimage)*0.1+0.85)
        #plt.colorbar(im1,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
        #ax1.set_title('YYYYY', fontsize=10)
        ax1.text(30, 65, 
            '''predictions: \n 
              happy {:.3f} \n
              neutral {:.3f} \n
              sad {:.3f}'''.format(predictions[0][0],predictions[0][1], 
              predictions[0][2]),
              ha="left", va="center",color='k' , fontsize=14)
        #plt.savefig(finalsavedirectory+'\\{}_{}_predictions.png'.format(category,image_number),dpi=100)
        

        # finally, apply the basic bubbles technique       
        bubbles_iterations=1000
        bubbles_acc_threshold= 0.5/predictions[0,correct_label]   #0.75
        bubbles_number=10
        bubbles_sigma=6.4
        tic=time.time()
        correct_plane, total_plane, bubbles_vec, acc_vec, relative_acc_threshold = vis_funcs.bubbles_technique_2(iterations=bubbles_iterations, 
                                                                             model=model, 
                                                                             imagelist=imagePMK, 
                                                                             bubbles=bubbles_number, 
                                                                             sigma=bubbles_sigma, 
                                                                             img_label_idx=correct_label, 
                                                                             acc_baseline=predictions[0,correct_label], # the predictions of the correct label, not the label with the maximum likelihood
                                                                             acc_threshold=bubbles_acc_threshold)
        toc=time.time()-tic
        print('bubbles: {:.2f} seconds'.format(toc))

        
        # convert total_plane and correct_plane from lists to normalized images
        correctplane_img=vis_funcs.bubbles_correct_plane(correct_plane)
        totalplane_img=vis_funcs.bubbles_total_plane(total_plane)
        
        # acc_vec contains all the classification results across trials, we are only interested in the vector of the correct label
        real_acc_vec=[]
        for acc in acc_vec:
            real_acc_vec.append(acc[0,correct_label])
        
        # remove outliers from the images, typically in the corners
        totalplane_img=vis_funcs.bubbles_remove_img_outliers(totalplane_img, std_limit_down=2, std_limit_up=4)
        correctplane_img=vis_funcs.bubbles_remove_img_outliers(correctplane_img, std_limit_down=2, std_limit_up=4)

        
        
        #====================================================================
        fig, axes = plt.subplots(2,2, figsize=(13, 10))       
        axes[0,0].axis('off')
        im1=axes[0,0].imshow(vis_funcs.normalize(baseimage))   
        im2=axes[0,0].imshow(vis_funcs.normalize(correctplane_img/(totalplane_img+1e-5)),alpha=0.55,cmap='jet')   
        axes[0,0].set_title('Correct_Plane/Total_Plane ',fontsize=12)
        fig.colorbar(im2,ax=axes[0,0],ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical') 
        axes[0,1].axis('off')
        im1=axes[0,1].imshow(vis_funcs.normalize(baseimage))                   
        axes[1,0].axis("off")
        axes[1,0].set_title('Image: {} {}'.format(category,image_number), fontsize=14)
        axes[1,0].imshow(vis_funcs.normalize(baseimage)*0.2+0.8)
        axes[1,0].text(15, 65, 
            '''
            category: {}\n
            Acc baseline: {:.2f}\n
            Iterations: {}\n
            sigma: {}\n
            Acc Threshold: {}\n
            Mean bubbles: {:.2f}\n
            '''.format(category,predictions[0,correct_label], bubbles_iterations, bubbles_sigma,
              str(int(relative_acc_threshold*100))+'%', np.mean(bubbles_vec)),
              ha="left", va="center",color='k' , fontsize=14)
        axes[1,1].hist(real_acc_vec)
        axes[1,1].axvline(x=predictions[0,correct_label], color='r',linestyle='--', label='baseline accuracy \ncategory: {}'.format(category))
        axes[1,1].axvline(x=relative_acc_threshold*predictions[0,correct_label], color='b',linestyle='--', label='Acc Threshold'.format(category))
        axes[1,1].legend()
        axes[1,1].set_xlabel('Accuracy')
        axes[1,1].set_ylabel('Count')        
        axes[1,1].set_title('Histogram of accuracy across Bubbles trials ')
        plt.suptitle('Full BUBBLES procedure \n Image: {} {}'.format(category,image_number), fontsize=15)  
        plt.savefig(finalsavedirectory+'\\{}_{}_full_BUBBLES.png'.format(category,image_number),dpi=100)
        plt.show()     

        #====================================================================
        plt.plot(bubbles_vec)
        plt.show()
        
        '''
        #====================================================================
        fig, axes = plt.subplots(3,3, figsize=(13, 13)) 
        for cols in range(3):
            for rows in range(3):
                idx=np.random.randint(0,len(total_plane))
                axes[rows,cols].axis('off')
                mask=total_plane[idx]
                mask=np.repeat(mask[:,:,np.newaxis],3,-1)
                axes[rows,cols].imshow(vis_funcs.normalize(baseimage)*mask)   
                axes[rows,cols].set_title('mask {}'.format(idx),fontsize=12)
        plt.suptitle('Random Bubbles masks \n Image: {} {}'.format(category,image_number), fontsize=15) 
        plt.savefig(finalsavedirectory+'\\{}_{}_random_bubbles_MASKS.png'.format(category,image_number),dpi=100)  

        #====================================================================
        '''
end=time.time()-start 
print('... \n ===== {:.2f} minutes ===='.format(end/60))    





#%%
#=================   Confusion Matrix of the model with test dataset =========


from sklearn.metrics import classification_report, confusion_matrix


evaluatedmodel=model.evaluate_generator(generator=testgenerator,steps=testsetsize/batch_size) #batch_size and testsetsize must be defined previously, generally in the script with the model
toc = time.time()-tic
print('Loss {}, Accuracy {}'.format(evaluatedmodel[0], evaluatedmodel[1]))
print('elapsed time: {} hr/min/sec'.format (printelapsedtime(toc)))






#%% ===========================================================================
#   ===========================================================================
#   Copied from CFA_evaluation_4AttModules.py


AttModules=4

if AttModules==3:
    layer_names=['reshape_2','reshape_5','reshape_8']
    layer_titles=['uSGS_01', 'lSGS_01','SO']
elif AttModules==4:
    layer_names=['reshape_2','reshape_5','reshape_8','reshape_11']
    layer_titles=['uSGS_01', 'lSGS_01','lSGS_02','SO']
elif AttModules==5:
    layer_names=['reshape_2','reshape_5','reshape_8','reshape_11','reshape_14']
    layer_titles=['uSGS_01', 'uSGS_02', 'lSGS_01','lSGS_02','SO']
    
    


savedirectory='SaliencyMaps'

start=time.time() 
for category, imagelist in sorted(imagesdict.items()):
    print('category: {}'.format(category))
    for image_number in imagelist:
        print('working on image: {}  {}'.format(category, image_number))
        
        #check if savedirectory exists otherwise create it
        finalsavedirectory=pathtomodel+'\\'+savedirectory+'\\'+category
        if not os.path.exists(finalsavedirectory):
            os.makedirs(finalsavedirectory)
        
# load base image============================================================== 
        baseimage = skimage.io.imread(basepath + '\\' + dataset_baseimage + '\\' + train_or_test + '\\' +  category + '\\' + train_or_test + '_' + category + '_' + image_number + '.jpg')
        baseimage=resize(baseimage, (128,128),preserve_range=True, anti_aliasing=True)
        #vis_funcs.display_images([baseimage], cols=2)

        
        # load PMK streams images as a list of tensors
        # imagePMK list of three tensors shape:128,128,3
        imagePMK=vis_funcs.load_imagePMK(image_number, category, 
                           image_size=(128,128),
                           dataset=dataset, 
                           basepath=basepath, 
                           train_or_test=train_or_test )

# visualize Base Image next to the PMK streams=================================
        imagesAll=[baseimage]
        imagesAll.extend(imagePMK)
        vis_funcs.display_images(imagesAll, cols=4, titles=['Base Image', 'P', 'M', 'K'])
        plt.savefig(finalsavedirectory+'\\{}_{}_base.png'.format(category,image_number),dpi=100)
                
# Preprocess imagePMK and get the predictions for the imagePMK ================
    #convert images to float32
        imagePMK=[image.astype(np.float32) for image in imagePMK]
    # Make it a batch of one. The model expects a batch, not a single image
        imagePMK=[image[np.newaxis,...] for image in imagePMK]
    #remember tha the input tensor needs to be [0,1]
        imagePMK=[image/255. for image in imagePMK]

        predictions =vis_funcs.predict3emotions(model, imagePMK)
        
        label_index = np.argmax(predictions) #useful for later, the label of the max category
    
    
    
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10, 5))
        ax0.axis("off")
        ax0.set_title('Image: {} {}'.format(category,image_number), fontsize=14)
        ax0.imshow(vis_funcs.normalize(baseimage), interpolation=None)
        #=====
        ax1.axis("off")
        ax1.imshow(vis_funcs.normalize(baseimage)*0.1+0.85)
        #plt.colorbar(im1,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
        #ax1.set_title('YYYYY', fontsize=10)
        ax1.text(30, 65, 
            '''predictions: \n 
              happy {:.3f} \n
              neutral {:.3f} \n
              sad {:.3f}'''.format(predictions[0][0],predictions[0][1], 
              predictions[0][2]),
              ha="left", va="center",color='k' , fontsize=14)
        plt.savefig(finalsavedirectory+'\\{}_{}_predictions.png'.format(category,image_number),dpi=100)
        
        

    
#==============================================================================    
# Attention Layer Activations =================================================   
        # get all attention masks and combine them in a single one    
        #layer_names=['reshape_2','reshape_5','reshape_8','reshape_11'] #,'reshape_14' #!!!!!!!!! (always check the model's graph plot in order to be sure which)
        masks=[]
        for layer in layer_names:
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            mask=np.sum(outputs, axis=-1) #sum all the masks from a single AttModule to get a single 2D mask
            mask=mask.squeeze()
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            masks.append(mask)
        masks=np.stack(masks) # stack the resulting masks from the AttModules
                 
        mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
        mask=vis_funcs.normalize(mask)
         
 
        #overlay of mask on image with side colorbar
        fig=plt.figure(figsize=(6,4))    
        plt.axis('off')
        im1=plt.imshow(vis_funcs.normalize(baseimage))   
        im2=plt.imshow(mask,alpha=0.7,cmap='jet')   
        plt.title('Salience Overlay V1\n Image: {} {}'.format(category,image_number),fontsize=12)
        plt.colorbar(im2,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
        plt.savefig(finalsavedirectory+'\\{}_{}_Attention_overlay_V1.png'.format(category,image_number),dpi=100)
        plt.show()



        
        #titles=['uSGS_01', 'lSGS_01','lSGS_02','SO'] #'uSGS_02',
        fig, axes = plt.subplots(2,3, figsize=(15, 10))
        # overall composite mask over image
        axes[0,0].axis("off")
        axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
        axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
        axes[0,0].set_title('Total Salience map V1', fontsize=14, color='r')  
        for n,layer in enumerate(layer_names):
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            outputs[outputs<.7]=0 #to avoid problems when doing boolean operations
            mask=np.sum(outputs, axis=-1) #sum all the masks from a single AttModule to get a single 2D mask  
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            i=0 if n<2 else 1
            j=n+1 if n<2 else n-2
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
            axes[i,j].axis("off")
            #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
            axes[i,j].set_title(layer_titles[n], fontsize=14)               
            axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
        plt.suptitle('Salience Maps V1 \n Image: {} {}'.format(category,image_number))    
        plt.savefig(finalsavedirectory+'\\{}_{}_SalienceMaps_by_SClayer.png'.format(category,image_number),dpi=100)
        plt.show()     





#======= SECOND WAY TO COMBINE MASKS!!!!!
        # get all attention masks and combine them in a single one    
        #layer_names=['reshape_2','reshape_5','reshape_8','reshape_11']  #,'reshape_14' 
        masks=[]
        for layer in layer_names:
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            outputs[outputs<.7]=0 #threshold, useful for the first layers, pay attention
            mask=np.any(outputs, axis=-1) #Logical OR across last dimension
            mask=mask.squeeze()
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            masks.append(mask)
        masks=np.stack(masks) # stack the resulting masks from the AttModules
        mask=np.sum(masks, axis=0) #sum the masks from the AttModules and get a single mask for ALL the AttModules
        mask=vis_funcs.normalize(mask) #normaize to [0,1]


        #overlay of mask on image with side colorbar
        fig=plt.figure(figsize=(6,4))    
        plt.axis('off')
        im1=plt.imshow(vis_funcs.normalize(baseimage))   
        im2=plt.imshow(mask,alpha=0.7,cmap='jet')   
        plt.title('Salience Overlay V2 \n Image: {} {}'.format(category,image_number),fontsize=12)
        plt.colorbar(im2,ticks=[0.1,0.3,0.5,0.7,0.9], orientation ='vertical')
        plt.savefig(finalsavedirectory+'\\{}_{}_Attention_overlay_V2.png'.format(category,image_number),dpi=100)
        plt.show()


        #titles=['uSGS_01', 'lSGS_01','lSGS_02','SO']# 'uSGS_02',
        fig, axes = plt.subplots(2,3, figsize=(15, 10))
        # overall composite mask over image (previously calculated)
        axes[0,0].axis("off")
        axes[0,0].imshow(vis_funcs.normalize(baseimage)) 
        axes[0,0].imshow(mask,alpha=0.7,cmap='jet')
        axes[0,0].set_title('Total Salience map V2', fontsize=14, color='r')  
        #calcuate the rest
        for n,layer in enumerate(layer_names):
            outputs=vis_funcs.read_layer(model, imagePMK, layer)
            outputs[outputs<.7]=0 #to avoid problems when doing boolean operations
            mask=np.any(outputs, axis=-1)
            mask = skimage.transform.resize(vis_funcs.normalize(mask), (128,128), order=1).copy()
            i=0 if n<2 else 1
            j=n+1 if n<2 else n-2
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')            
            axes[i,j].axis("off")
            #axes[i,j].set_title('i{} j{} n{}'.format(i,j,n), fontsize=14)  
            axes[i,j].set_title(layer_titles[n], fontsize=14)               
            axes[i,j].imshow(vis_funcs.normalize(baseimage)) 
            axes[i,j].imshow(mask,alpha=0.7,cmap='jet')
        plt.suptitle('Salience Maps V2 \n Image: {} {}'.format(category,image_number))    
        plt.savefig(finalsavedirectory+'\\{}_{}_SalienceMaps_by_SClayer_2ndProcessing.png'.format(category,image_number),dpi=100)
        plt.show()   


end=time.time()-start 
print('... \n ===== {:.2f} minutes ===='.format(end/60))     







    
        
        
        
  
             
#%%  GARBAGE ZONE
        
        xxx=vis_funcs.bubbles_mask_tensor(vis_funcs.normalize(baseimage),rowscols=4,gaussian=True, sigma=20)
        



        rowscols=4
        imagePMK_bubbles=[vis_funcs.bubbles_mask_tensor(image,rowscols=rowscols,gaussian=True, sigma=15) for image in imagePMK]
        bubbles_y = model.predict(imagePMK_bubbles)       
        vis_funcs.display_images(imagePMK_bubbles[0])
        
        
        probs = bubbles_y[:, 3] #choose the label index obtained with the classification of the complete image
        vis_funcs.tensor_summary(probs) 
        fig=plt.figure(figsize=(8,5)) 
        plt.hist(probs)
        plt.title('Bubbles: Histogram of classification probabilities \n Image: {} {}'.format(category,image_number))

            
           # normalize probabilities [0,1] to construct the mask
        mask=[]
        mask = vis_funcs.normalize(probs)
        mask = np.reshape(mask, (rowscols, rowscols)) #reshape probs vector
        vis_funcs.tensor_summary(mask)    

        probs_matrix = np.reshape(probs, (rowscols, rowscols)) #reshape the probs vector (original values, no normalization like before)
           
        vis_funcs.apply_mask_heatmap(baseimage, mask, probs_matrix, order=0, cmap='OrRd',
                                     title1='Image: {} {}'.format(category,image_number))   
    
        plt.imshow(imagePMK_bubbles[0][np.argsort(probs)[-1],...])
        plt.imshow(imagePMK_bubbles[0][np.argsort(probs)[-2],...])
       plt.imshow(imagePMK_bubbles[0][np.argsort(probs)[0],...])
        
        
        
        
        
        
        
        
x, y = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.3, 0.0
g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
plt.imshow(g)        
    























            