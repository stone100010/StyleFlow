def psp(image_path = None,  net = None,  opts = None):
    from argparse import Namespace
    import time
    import os
    import sys
    import pprint
    import numpy as np
    from PIL import Image
    import torch
    import torchvision.transforms as transforms

    sys.path.append(".")
    sys.path.append("..")

    from datasets import augmentations
    from utils.common import tensor2im, log_input_image
    from models.psp import pSp
    import numpy as np
    from sklearn.manifold import TSNE

    import os

    CODE_DIR = 'pixel2style2pixel'

    experiment_type = 'ffhq_encode'

    def get_download_model_command(file_id, file_name):
        """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
        current_directory = os.getcwd()
        save_path = os.path.join(os.path.dirname(current_directory), CODE_DIR, "pretrained_models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
        return url
        
        
    MODEL_PATHS = {
        "ffhq_encode": {"id": "1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "name": "psp_ffhq_encode.pt"},
        "ffhq_frontalize": {"id": "1_S4THAzXb-97DbpXmanjHtXRyKxqjARv", "name": "psp_ffhq_frontalization.pt"},
        "celebs_sketch_to_face": {"id": "1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA", "name": "psp_celebs_sketch_to_face.pt"},
        "celebs_seg_to_face": {"id": "1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz", "name": "psp_celebs_seg_to_face.pt"},
        "celebs_super_resolution": {"id": "1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu", "name": "psp_celebs_super_resolution.pt"},
        "toonify": {"id": "1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz", "name": "psp_ffhq_toonify.pt"}
    }


    path = MODEL_PATHS[experiment_type]
    download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])


    EXPERIMENT_DATA_ARGS = {
        "ffhq_encode": {
            "model_path": "pretrained_models/psp_ffhq_encode.pt",
            "image_path": "/home/dibabdal/Desktop/MySpace/Devs/SfSNet-Pytorch-master/Images/REF_ID00353_Cam11_0063.png",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        },
        "ffhq_frontalize": {
            "model_path": "pretrained_models/psp_ffhq_frontalization.pt",
            "image_path": "notebooks/images/input_img.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        },
        "celebs_sketch_to_face": {
            "model_path": "pretrained_models/psp_celebs_sketch_to_face.pt",
            "image_path": "notebooks/images/input_sketch.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()])
        },
        "celebs_seg_to_face": {
            "model_path": "pretrained_models/psp_celebs_seg_to_face.pt",
            "image_path": "notebooks/images/input_mask.png",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.ToOneHot(n_classes=19),
                transforms.ToTensor()])
        },
        "celebs_super_resolution": {
            "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
            "image_path": "notebooks/images/input_img.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.BilinearResize(factors=[16]),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        },
        "toonify": {
            "model_path": "pretrained_models/psp_ffhq_toonify.pt",
            "image_path": "notebooks/images/input_img.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        },
    }
    
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    
    if net is None:
        


        model_path = EXPERIMENT_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cpu')


        opts = ckpt['opts']
        # update the training options
        opts['checkpoint_path'] = model_path
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
            
        opts= Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        print('Model successfully loaded!')

    if image_path is None:
        image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
        
    original_image = Image.open(image_path)
    if opts.label_nc == 0:
        original_image = original_image.convert("RGB")
    else:
        original_image = original_image.convert("L")

    original_image.resize((256, 256))


    def run_alignment(image_path):
      import dlib
      from scripts.align_all_parallel import align_face
      predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
      aligned_image = align_face(filepath=image_path, predictor=predictor) 
      print("Aligned image has shape: {}".format(aligned_image.size))
      return aligned_image
      
    if experiment_type not in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
      input_image = run_alignment(image_path)
    else:
      input_image = original_image

    input_image.resize((256, 256))
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    def run_on_batch(inputs, net, latent_mask=None):
        latent = None
        if latent_mask is None:
            result_batch,  latent = net(inputs.to("cuda").float(), randomize_noise=False,  return_latents = True)
            
        else:
            result_batch = []
            for image_idx, input_image in enumerate(inputs):
                # get latent vector to inject into our input image
                vec_to_inject = np.random.randn(1, 512).astype('float32')
                _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                          input_code=True,
                                          return_latents=True)
                
                # get output image with injected style vector
                res = net(input_image.unsqueeze(0).to("cuda").float(),
                          latent_mask=latent_mask,
                          inject_latent=latent_to_inject)
                result_batch.append(res)
            result_batch = torch.cat(result_batch, dim=0)
        return result_batch,  latent
        
    if experiment_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
        latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    else:
        latent_mask = None
        
    with torch.no_grad():
        tic = time.time()
        result_image,  latent = run_on_batch(transformed_image.unsqueeze(0), net, latent_mask)
        result_image = result_image[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    input_vis_image = log_input_image(transformed_image, opts)
    output_image = tensor2im(result_image)

    if experiment_type == "celebs_super_resolution":
        res = np.concatenate([np.array(input_image.resize((256, 256))),
                              np.array(input_vis_image.resize((256, 256))),
                              np.array(output_image.resize((256, 256)))], axis=1)
    else:
        res = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                              np.array(output_image.resize((256, 256)))], axis=1)
                              

    res_image = Image.fromarray(res)
    import gc
    gc.collect()
    return res_image,  latent,  net,  opts
    

def listImages():
    f = open("/home/dibabdal/Desktop/MySpace/Devs/styleFlow/StyleFlow/indices", "r")
    start = int(f.readline())
    limit_ = int(f.readline())    
    import os
    images = []
    import csv
    counter = 0
    limit = limit_ + start
    with open('/home/dibabdal/Desktop/MySpace/Devs/SfSNet-Pytorch-master/sipr_data2/all.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if counter < start:
                counter += 1
                continue
                
            if limit != -1 and counter >= limit:
                break
            r1 = '/home/dibabdal/Desktop/MySpace/Devs/SfSNet-Pytorch-master/sipr_data2/reference/' + row[1]
            r0 = '/home/dibabdal/Desktop/MySpace/Devs/SfSNet-Pytorch-master/sipr_data2/input/' + row[0]
            if os.path.isfile(r1) and os.path.isfile(r0):
                images.append(r1)
                images.append(r0)
            else:
                print('not found :',  row[0],  ' or ',  row[1])
            
            counter += 1
            #print(', '.join(row))
    
    return images
    
if __name__ == "__main__":
    from os import walk
    import pickle
    
    if False:
        raw_w = pickle.load(open("sg2latents.pickle", "rb"))
        print(raw_w['Latent'][0])
        exit(0)

    #path = "/home/dibabdal/Desktop/MySpace/Devs/SfSNet-Pytorch-master/Images/"
    filenames = listImages()
    
    latents = []
    net = None
    opts = None
    for i in range(len(filenames)): #
        try:
            print("processing: ",  filenames[i])
            image,  latent,  net,  opts = psp(filenames[i],  net,  opts)
            latents.append(latent.cpu().numpy())
            print('latent shape=>',  latent.shape)
            #image.save("output_" + str(i) + ".png")
        except:
            print("exception occured")
            import numpy as np
            zeros = np.zeros([1,  18,  512])
            latents.append(zeros)
       
            
    workingDir = '/home/dibabdal/Desktop/MySpace/Devs/styleFlow/StyleFlow/data/'
    with open(workingDir + '/sg2latents.pickle','wb') as f:
        pickle.dump( {'Latent' : latents}, f)
    
    import numpy as np
    tsne = np.load("/home/dibabdal/Desktop/MySpace/Devs/styleFlow/StyleFlow/dataLegacy/TSNE.npy")
    x = tsne[0:len(latents), :]
    
    workingDir = '/home/dibabdal/Desktop/MySpace/Devs/styleFlow/StyleFlow/data/'
    np.save(workingDir + "/TSNE",  x)

    #
    #print(np.array(latents).shape)


        
    
    
