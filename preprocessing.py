from math import log , exp

def log_mean(a,b):
    return exp((log(a)+log(b))/2)

def center_around_zero(tens):
    tens = tens - torch.mean(tens)/2



def clamp_and_normalise_exr_tensor(tens):
    """
    Returns a normalised version of an image tensor comming from an un-clamped source.
    To do that we choose a max treshold of light intensity, clamp the image and the normalise
    """

    max = torch.max(tens)
    mean = torch.mean(tens)
    median = torch.median(tens)
    magic = log_mean(max,mean)
    # print(f"max {max}")
    # print(f"mean {mean}")
    # print(f"median {median}")
    # print(f"magic {magic}")

    tensMax = tens/max
    tensMean = tens/mean
    tensMedian = tens/median
    tensMagic = tens/magic

    l1 = torch.cat((tensMax,tensMean),3)
    l2 = torch.cat((tensMagic,tensMedian),3)
    show = torch.cat((l1,l2),2)
    show = (show /255) -0.5 
    
    showImage( show  ,"float32")
    


if __name__ == "__main__":
    from dataManager import *
    from torch.utils.data import DataLoader

    folder = "./generated_images_512_train"
    data = MyDataset(folder)
    data.check_data()
    train_dataloader = DataLoader(data, batch_size=1, shuffle=True)
    for i in range(1):
        
        in_tens, out_tens, ext_tens = next(iter(train_dataloader))

        in_tens = (in_tens +0.5) * 255 


        print(in_tens.size())
        dic = data.separate_into_images(out_tens)

        clamp_and_normalise_exr_tensor(in_tens)

        
        # showImage(ext_tens,"float32")#exr images need to be seen in float32 for correct exposure
        # showImage(dic["base"])
     

