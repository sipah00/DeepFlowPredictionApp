import numpy as np
from PIL import Image
from matplotlib import cm

class InputData:
    def __init__(self, npz_arr, removePOffset=True, makeDimLess=True):
        self.input = None
        self.target = None

        self.max_inputs_0 = 100.0
        self.max_inputs_1 = 38.12
        self.max_inputs_2 = 1.0

        self.max_targets_0 = 4.65 
        self.max_targets_1 = 2.04
        self.max_targets_2 = 2.37

        if npz_arr.shape[0] >= 3:
            self.input = npz_arr[0:3]
        if npz_arr.shape[0] == 6:
            self.target = npz_arr[3:6]

        self.removePOffset = removePOffset
        self.makeDimLess = makeDimLess

        self.normalize()

    def normalize(self):
        if self.target is not None:
            if self.removePOffset:
                self.target[0,:,:] -= np.mean(self.target[0,:,:]) # remove offset
                self.target[0,:,:] -= self.target[0,:,:] * self.input[2,:,:]  # pressure * mask

            if self.makeDimLess: 
                v_norm = ( np.max(np.abs(self.input[0,:,:]))**2 + np.max(np.abs(self.input[1,:,:]))**2 )**0.5 
                self.target[0,:,:] /= v_norm**2
                self.target[1,:,:] /= v_norm
                self.target[2,:,:] /= v_norm

            self.target[0,:,:] *= (1.0/self.max_targets_0)
            self.target[1,:,:] *= (1.0/self.max_targets_1)
            self.target[2,:,:] *= (1.0/self.max_targets_2)

        if self.input is not None:
            self.input[0, :, :] *= 1 / self.max_inputs_0
            self.input[1, :, :] *= 1 / self.max_inputs_1

    def denormalize(self, data, v_norm):
        a = data.copy()
        a[0,:,:] /= (1.0/self.max_targets_0)
        a[1,:,:] /= (1.0/self.max_targets_1)
        a[2,:,:] /= (1.0/self.max_targets_2)

        if self.makeDimLess:
            a[0,:,:] *= v_norm**2
            a[1,:,:] *= v_norm
            a[2,:,:] *= v_norm
        return a


# image output
def imageOut(filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
    outputs = np.copy(_outputs)
    targets = np.copy(_targets)
    
    s = outputs.shape[1] # should be 128
    if saveMontage:
        new_im = Image.new('RGB', ( (s+10)*3, s*2) , color=(255,255,255) )
        BW_im  = Image.new('RGB', ( (s+10)*3, s*3) , color=(255,255,255) )

    for i in range(3):
        outputs[i] = np.flipud(outputs[i].transpose())
        targets[i] = np.flipud(targets[i].transpose())
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        if normalize:
            outputs[i] -= min_value
            targets[i] -= min_value
            max_value -= min_value
            outputs[i] /= max_value
            targets[i] /= max_value
        else: # from -1,1 to 0,1
            outputs[i] -= -1.
            targets[i] -= -1.
            outputs[i] /= 2.
            targets[i] /= 2.

        if not saveMontage:
            suffix = ""
            if i==0:
                suffix = "_pressure"
            elif i==1:
                suffix = "_velX"
            else:
                suffix = "_velY"

            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            im = im.resize((512,512))
            im.save(filename + suffix + "_pred.png")

            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            if saveTargets:
                im = im.resize((512,512))
                im.save(filename + suffix + "_target.png")

        if saveMontage:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*0))
            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*1))

            im = Image.fromarray(targets[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*0))
            im = Image.fromarray(outputs[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*1))
            imE = Image.fromarray( np.abs(targets[i]-outputs[i]) * 10.  * 256. )
            BW_im.paste(imE, ( (s+10)*i, s*2))

    if saveMontage:
        new_im.save(filename + ".png")
        BW_im.save( filename + "_bw.png")


def saveOutput(output_arr, target_arr):
    imageOut("result", output_arr, target_arr, normalize=False, saveMontage=True) # write normalized with error
