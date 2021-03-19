
import os, sys
import numpy as np

from os import walk
import pickle



def psp_useful_npy(npyfilenames, output_path):

    latents = []
    net = None
    opts = None
    for i in range(len(npyfilenames)):  #
        try:
            print("processing: ", npyfilenames[i])
            image, latent, net, opts = psp(npyfilenames[i], net, opts)
            latents.append(latent.cpu().numpy())
            print('latent shape=>', latent.shape)
            # image.save("output_" + str(i) + ".png")
        except:
            print("exception occured")
            import numpy as np

            zeros = np.zeros([1, 18, 512])
            latents.append(zeros)

    workingDir = '/home/dibabdal/Desktop/MySpace/Devs/styleFlow/StyleFlow/data/'
    with open(workingDir + '/sg2latents.pickle', 'wb') as f:
        pickle.dump({'Latent': latents}, f)

    import numpy as np

    tsne = np.load("/home/dibabdal/Desktop/MySpace/Devs/styleFlow/StyleFlow/dataLegacy/TSNE.npy")
    x = tsne[0:len(latents), :]

    np.save(output_path + "/TSNE", x)

    #
    # print(np.array(latents).shape)

def npy_list(inp, outp, inp_tyle):
    num = 0
    for root, dirs, files in os.walk(inp):
        for file in files:
            file_name, file_type = file.split(".")
            if file_type == inp_tyle:
                real_file_path = os.path.join(root, file)

                num = num + 1
                print(num, real_file_path)


if __name__ == '__main__':
    npy_path = '../newimg_data/'
    npy_list(npy_path, npy_path, "npy")
