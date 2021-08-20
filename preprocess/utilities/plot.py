import matplotlib.pyplot as plt 
import matplotlib.patches as patches

def plot_with_bbox(image, fv, size_correction = None):
    """
        Graph the bounding boxes on an image

        Take the feature vectors, corresponding to the 
        element class and parameters of the bounding box 
        and plot it 

        Parameters
        ----------
        image : np.array
                3 - Channel image numpy array. 
        
        fv : np,array
                Feature vector containing n - bounding boxes 
                with 5 elements (1 class  + 4 parameters)
        
    """
    print(fv)
    fig, ax = plt.subplots()
    for feature_vector in fv:
    # print(feature_vector[1])
        [x, y, w, h] = feature_vector[-4:]
        
        plt.imshow(image * 255)
    
        (IH, IW) = image.shape[0:2]

        w = IW * w
        h = IH * h

        x = (IW * x) - w/2
        y = (IH * y) - h/2

        # Prepare the rect 
        rect = patches.Rectangle((x, y), w,  h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

    