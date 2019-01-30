'''Create line art by linking points around unit circle.
'''




IMG_FILE='people05.jpg'
HOUGH_FILE='hough_data.npy'

IMG_WIDTH=300  # width of image to resize to
N_POINTS=100   # NO of points on circle
SIGMA=2        # Gaussian blur size
LINE_WIDTH=2.4 # max line width
DISK=2
ANIMATE=True  # if True, create animation, a final plot otherwise
NLAYERS=10
MIN_RATIO=0.5




#--------Import modules-------------------------
import numpy as np
from PIL import Image
from PIL import ImageFilter
from skimage import filters
import plot
import matplotlib.pyplot as plt
from skimage.transform import hough_line_peaks as hlp





def getLineByPoints(p1,p2,verbose=True):
    '''Get a line between 2 points on a cartisian grid.

    <p1>, <p2>: (x,y) coordinates of 2 points.

    Return <result>: Nx2 array, coordinates linking <p1> to <p2> with a 
                     straight line.

    See also getLine(): get a line by theta and length.
    '''

    import numpy as np

    #-------------------Check inputs-------------------
    p1=np.asarray(p1).squeeze()
    p2=np.asarray(p2).squeeze()

    if p1.ndim!=1 or len(p1)!=2:
        raise Exception("<p1> needs to be length-2 1d array.")
    if p2.ndim!=1 or len(p2)!=2:
        raise Exception("<p2> needs to be length-2 1d array.")

    x1,y1=p1
    x2,y2=p2

    if np.all(p1==p2):
        result=np.asarray(p1)[None,:]
        #result=p1
    elif x1==x2 and y1!=y2:
        step=np.sign(y2-y1)
        ys=np.arange(y1,y2+step,step)
        result=np.c_[x1*np.ones(len(ys)), ys]
        #result=zip(ys, [x1,]*len(ys))
    elif y1==y2 and x1!=x2:
        step=np.sign(x2-x1)
        xs=np.arange(x1,x2+step,step)
        result=np.c_[xs, y1*np.ones(len(xs))]
        #result=zip([y1,]*len(xs), xs)
    else:
        beta=float(y2-y1)/(x2-x1)
        if abs(beta)>=1:
            step=np.sign(y2-y1)
            ys=np.arange(y1,y2+step,step)
            xs=((ys-y1)/beta).astype('int')+x1
            result=np.c_[xs,ys]
            #result=zip(ys,xs)
        else:
            step=np.sign(x2-x1)
            xs=np.arange(x1,x2+step,step)
            ys=((xs-x1)*beta).astype('int')+y1
            result=np.c_[xs,ys]
            #result=zip(ys,xs)

    return result


def plotLines(xs,ys,widths):
    nn=0
    for ii in range(N_POINTS):
        for jj in range(ii,N_POINTS):
            if ii==jj:
                continue

            if widths[nn]==0:
                nn+=1
                continue

            ax.plot([xs[ii],xs[jj]], [ys[ii],ys[jj]],
                    linewidth=widths[nn],
                    color=str((1.-occratio[nn])/3.))
            nn+=1

    return

def getLines(xs,ys,widths):

    nn=0
    xsegs=[]
    ysegs=[]
    colorsegs=[]
    for ii in range(N_POINTS):
        for jj in range(ii,N_POINTS):
            if ii==jj:
                continue

            #ax.plot([xs[ii],xs[jj]], [ys[ii],ys[jj]],
                    #linewidth=widths[nn],
                    #color=str((1.-occratio[nn])/3.))
            xsegs.append([xs[ii],xs[jj]])
            ysegs.append([ys[ii],ys[jj]])
            colorsegs.append(str((1.-occratio[nn])/3.))
            nn+=1

    return xsegs,ysegs,colorsegs




#-------------Main---------------------------------
if __name__=='__main__':

    #--------------------Read image--------------------
    img=Image.open(IMG_FILE).convert('L')

    #-------------------Resize image-------------------
    img_size=img.size # (x,y)
    img_ratio=float(img_size[1])/img_size[0] # y/x

    newsize=[IMG_WIDTH, int(IMG_WIDTH*img_ratio)]
    img=img.resize(newsize,Image.ANTIALIAS)

    # blur
    img=img.filter(ImageFilter.GaussianBlur(SIGMA))

    # convert to array
    img=np.array(img)
    img=img[::-1,:]

    # invert image
    img=img.max()-img

    # thresholding image
    otsu=filters.threshold_otsu(img)
    img_mask=np.where(img>otsu,1,0)

    #------------------Clip by circle------------------
    R=np.min(img.shape)//2

    xx=np.arange(img.shape[1])
    yy=np.arange(img.shape[0])
    XX,YY=np.meshgrid(xx,yy)
    circle=np.where((XX-img.shape[1]//2)**2+(YY-img.shape[0]//2)**2<=R**2,1,0)
    img=img*circle
    img_mask=img_mask*circle

    layers=np.linspace(np.min(img),np.max(img),NLAYERS+1)
    img_layers=np.zeros(img.shape)
    ii=0

    for z1,z2 in zip(layers[:-1],layers[1:]):
        img_layers=np.where((img>=z1) & (img<=z2),ii,img_layers)
        ii+=1

    #img_layers=img



    print('Image size: %s' %str(img.shape))

    #---------Get anchor points around circle---------
    dtheta=2*np.pi/N_POINTS
    thetas=np.arange(N_POINTS)*dtheta
    xs=R*np.cos(thetas)+img.shape[1]//2
    ys=R*np.sin(thetas)+img.shape[0]//2

    #---Get allowed Hough parameter space by circle lines----
    #hough=np.load(HOUGH_FILE)
    circ_coords=np.where(circle>0)

    cy1=circ_coords[0].min()
    cy2=circ_coords[0].max()
    cx1=circ_coords[1].min()
    cx2=circ_coords[1].max()

    square=img_layers[cy1:cy2+1, cx1:cx2+1]
    squaremask=img_mask[cy1:cy2+1, cx1:cx2+1]

    A=np.zeros([square.size, (N_POINTS-1)*N_POINTS/2])

    nn=0
    from skimage import morphology
    disk=morphology.disk(DISK)

    lengths=np.zeros(A.shape[1])
    occ=np.zeros(A.shape[1])

    #fig,ax=plt.subplots()
    for ii in range(N_POINTS):
        for jj in range(ii,N_POINTS):
            if ii==jj:
                continue

            print('processing line %d' %nn)
            p1=[xs[ii]-cx1,ys[ii]-cy1]
            p2=[xs[jj]-cx1,ys[jj]-cy1]
            lineidx=getLineByPoints(p1,p2).astype('int')

            pidx=lineidx[:,1]*square.shape[1]+lineidx[:,0]
            tmp=np.zeros(A.shape[0])
            tmp[pidx]=1.
            tmp=tmp.reshape(square.shape)

            lengths[nn]=tmp.sum()
            occ[nn]=(tmp*squaremask).sum()
            tmp=morphology.dilation(tmp,disk)

            A[:,nn]=tmp.flatten()
            nn+=1

    occratio=occ.astype('float')/lengths

    b=img_layers[cy1:cy2+1, cx1:cx2+1].flatten()

    print('Sovling linear equations ...')
    fire=np.linalg.lstsq(A,b)[0]

    def func(xx,aa,bb):
        cost=(np.dot(aa,xx)-bb)**2
        return np.sum(cost)

    #from scipy import optimize

    #x0=np.random.random(A.shape[1])
    #bounds=zip(np.zeros(x0.shape), [None,]*len(x0))
    #fire=optimize.minimize(func,x0,args=(A,b),
            #method='L-BFGS-B',bounds=bounds)



    #-------------------Plot------------------------
    figure=plt.figure(figsize=(9,9),dpi=100)
    ax=figure.add_subplot(111)

    ax.set_xlim([img.shape[1]//2-R, img.shape[1]//2+R])
    ax.set_ylim([img.shape[0]//2-R, img.shape[0]//2+R])
    ax.set_axis_off()
    ax.set_aspect('equal')

    widths=np.where(fire<fire.mean(),0,fire)
    widths=widths/np.max(widths)*LINE_WIDTH

    print('Plotting lines ...')



    if ANIMATE:
        #import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation

        widths2=5*widths*occratio**0.5
        xsegs1,ysegs1,colorsegs1=getLines(xs,ys,widths2)

        #widths3=8*widths**1.7*occratio
        #xsegs2,ysegs2,colorsegs2=getLines(xs,ys,widths3)

        def update(ii,xsegs,ysegs,widsegs,csegs):

            print('update line: %d' %ii)
            if widsegs[ii]==0:
                return

            ax.plot(xsegs[ii], ysegs[ii],
                    linewidth=widsegs[ii],
                    color=csegs[ii])


        #anim=FuncAnimation(figure,update,frames=len(xsegs1)+len(xsegs2),
            #fargs=(xsegs1+xsegs2, ysegs1+ysegs2, np.r_[widths2,widths3],
                #colorsegs1+colorsegs2),
        anim=FuncAnimation(figure,update,frames=len(xsegs1),
                fargs=(xsegs1, ysegs1, widths2, colorsegs1),
                interval=5,
                repeat=True,
                blit=False)
        anim.save('%s_line_animation.mp4' %IMG_FILE,
                fps=150)

    else:

        widths2=5*widths*occratio**0.5
        plotLines(xs,ys,widths2)

        widths3=8*widths**1.7*occratio
        plotLines(xs,ys,widths3)

        figure.show()
