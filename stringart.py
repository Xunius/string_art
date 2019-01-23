'''Create line art by linking points around unit circle.
'''




IMG_FILE='people04.jpg'
HOUGH_FILE='hough_data.npy'

IMG_WIDTH=400  # width of image to resize to
N_POINTS=120   # NO of points on circle
SIGMA=2        # Gaussian blur size
EPS=0.005      # error associating a line
N_THETA=360    # size of angle dimension in Hough transform
LINE_WIDTH=0.9 # max line width
ANIMATE=False  # if True, create animation, a final plot otherwise




#--------Import modules-------------------------
import numpy as np
from PIL import Image
from PIL import ImageFilter
from skimage import filters
import matplotlib.pyplot as plt
from skimage.transform import hough_line_peaks as hlp




def houghLine(img,n_theta=180):
    '''Line Hough transform, nested loop version'''

    height,width=img.shape
    thetas=np.deg2rad(np.linspace(-90.,90.,n_theta))
    diag_len=np.ceil(np.sqrt(width**2+height**2)).astype('int')
    rhos=np.linspace(-diag_len,diag_len,diag_len*2)

    cos_t=np.cos(thetas)
    sin_t=np.sin(thetas)
    num_thetas=len(thetas)

    # prepare accumulator
    accumulator=np.zeros((2*diag_len,num_thetas))
    yids,xids=np.nonzero(img)

    # loop though pixels
    for ii in range(len(xids)):
        x=xids[ii]
        y=yids[ii]

        for tt in range(num_thetas):
            rhoidx=int(round(x*cos_t[tt]+y*sin_t[tt]))+diag_len
            #accumulator[rhoidx,tt]+=1
            accumulator[rhoidx,tt]+=img[y,x]

    return accumulator, thetas, rhos



def houghLine2(img,n_theta=180):

    height,width=img.shape
    thetas=np.deg2rad(np.linspace(-90.,90.,n_theta))
    diag_len=np.ceil(np.sqrt(width**2+height**2)).astype('int')
    rhos=np.linspace(-diag_len,diag_len,diag_len*2)

    cos_t=np.cos(thetas)
    sin_t=np.sin(thetas)
    num_thetas=len(thetas)

    # prepare accumulator
    accumulator=np.zeros((2*diag_len,num_thetas))
    yids,xids=np.nonzero(img)

    dtheta=thetas[1]-thetas[0]
    drho=rhos[1]-rhos[0]

    # Create a grid from (rho, theta)
    # delta between row_{i+1} and row_{i} = pi + drho
    # delta between column_{j+1} and column_{j} = dtheta
    # drho = rhos[1] - rhos[0], constant
    # dtheta = thetas[1] - thetas[0], constant

    jjs=dtheta*np.arange(len(thetas))   # column deltas
    iis=(np.pi+drho)*np.arange(len(rhos))  # row deltas
    grid=iis[:,None]+jjs[None,:]+rhos[0]+thetas[0]

    # flatten grid to form bin edges
    grid_f=grid.flatten()
    #grid_f=np.r_[grid_f,np.max(grid_f)+dtheta]
    grid_f=np.r_[grid_f-dtheta/2.,grid_f[-1]+dtheta/2.]

    nonzeroidx=yids*img.shape[1]+xids
    weights=img.flatten()
    weights=np.take(weights,nonzeroidx)
    weights=np.repeat(weights[:,None],num_thetas,axis=1)
    weights=weights.flatten()

    # compute rho_hats as x*cos(theta) + y*sin(theta)
    rho_hats=np.outer(xids,cos_t)+np.outer(yids,sin_t)

    # convert rho_hats by adding row/column deltas
    rho_hats1=(np.pi+drho)*(rho_hats+diag_len).astype('int')
    rho_hats2=rhos[0]+thetas[None,:]+rho_hats1

    #accumulator=np.histogram(rho_hats2.flatten(),bins=grid_f)[0]
    accumulator=np.histogram(rho_hats2.flatten(),bins=grid_f,weights=weights)[0]
    accumulator=accumulator.reshape((len(rhos),len(thetas)))

    return accumulator, thetas, rhos


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
    otsu=filters.threshold_otsu(img)*0.8

    #from skimage import feature
    #img=feature.canny(img)

    img_mask=np.where(img>otsu,1,0)
    img=img*img_mask
    #img=img_mask
    img=img.astype('float')/img.max()

    print('Image size: %s' %str(img.shape))

    #------------------Clip by circle------------------
    R=np.min(img.shape)//2

    xx=np.arange(img.shape[1])
    yy=np.arange(img.shape[0])
    XX,YY=np.meshgrid(xx,yy)
    circle=np.where((XX-img.shape[1]//2)**2+(YY-img.shape[0]//2)**2<=R**2,1,0)
    img=img*circle

    #-----------------Hough transform-----------------
    print('Computing Hough transformation ...')
    accumulator, h_thetas, h_rhos=houghLine2(img,N_THETA)
    #accumulator2, h_thetas2, h_rhos2=houghLine(img,N_THETA)

    #---------Get anchor points around circle---------
    dtheta=2*np.pi/N_POINTS
    thetas=np.arange(N_POINTS)*dtheta
    xs=R*np.cos(thetas)+img.shape[1]//2
    ys=R*np.sin(thetas)+img.shape[0]//2

    #---Get hough parameters for all possible lines---
    A=xs[None,:]-xs[:,None]
    B=ys[None,:]-ys[:,None]

    alphas=np.arctan(-A/B)
    alphas=np.where(np.isnan(alphas),0,alphas)
    alphas=np.triu(alphas)

    rs=xs*np.cos(alphas)+ys*np.sin(alphas)

    #------------Get Hough parameter space------------
    n_r=accumulator.shape[0]
    n_alpha=accumulator.shape[1]
    h_alpha=np.linspace(-np.pi/2.,np.pi/2.,n_alpha)
    h_r=np.linspace(-1,1,n_r)*h_rhos.max()
    hough=np.zeros([n_r,n_alpha])
    scale_r=np.ptp(h_r)/2.
    scale_alpha=np.ptp(h_alpha)/2.

    alpha_m,r_m=np.meshgrid(h_alpha,h_r)

    #---Get allowed Hough parameter space by circle lines----
    try:
        hough=np.load(HOUGH_FILE)
    except:
        for ii in range(N_POINTS):
            for jj in range(ii,N_POINTS):
                if ii==jj:
                    continue
                alphaij=alphas[ii,jj]
                rij=rs[ii,jj]

                print('Get Hough parameter for line (%d,%d), alphaij = %.2f, rij = %.2f'\
                        %(ii,jj,alphaij,rij))

                houghij=(alpha_m-alphaij)**2/scale_alpha**2+(r_m-rij)**2/scale_r**2
                accij=np.where(houghij<=EPS**2,1,0)
                hough+=accij

    hough_masked=(hough>0)*accumulator


    from skimage.transform import probabilistic_hough_line as phl

    lines=phl(img,line_length=20,line_gap=5,
            theta=np.linspace(-np.pi/2,np.pi/2,n_alpha*2))
            #theta=h_alpha)

    lens=[]
    alist=[]
    blist=[]
    line_alphas=[]
    line_rs=[]
    intensities=[]


    fig2,ax=plt.subplots()
    for lii in lines:
        p0,p1=lii
        lineidx=getLineByPoints(p0,p1).astype('int')
        intii=img[lineidx[:,1],lineidx[:,0]]
        meanintii=np.mean(intii)

        if p1[0]==p0[0]:
            alphaii=0
            rii=p0[0]
        else:
            bii=float(p1[1]-p0[1])/(p1[0]-p0[0])
            aii=p0[1]-bii*p0[0]
            if bii!=0:
                alphaii=np.arctan(-1./bii)
                rii=aii*np.sin(alphaii)
            else:
                alphaii=np.pi/2.
                rii=aii

        lens.append((p0[0]-p1[0])**2+(p0[1]-p1[1])**2)
        #alist.append(aii)
        #blist.append(bii)
        line_alphas.append(alphaii)
        line_rs.append(rii)
        intensities.append(meanintii)

        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], color='Gray')

    lens=np.array(lens)
    sortidx=np.argsort(-lens)
    lens=np.sort(lens)[::-1].astype('float')

    lines=[lines[ii] for ii in sortidx]
    line_alphas=[line_alphas[ii] for ii in sortidx]
    line_rs=[line_rs[ii] for ii in sortidx]
    intensities=[intensities[ii] for ii in sortidx]

    fig2.show()

    #maxh=np.max(intensities)
    #peaks=zip(intensities,line_alphas,line_rs)
    peaks=zip(lens,line_alphas,line_rs)
    maxh=np.max(lens)

    #------------Get Hough parameter peaks------------
    #maxh=hough_masked.max()
    #peaks=hlp(accumulator,h_thetas,h_rhos,min_distance=1,min_angle=1,
            #threshold=maxh/8.)
    #peaks=zip(*peaks)
    print('len(peaks) = %d' %len(peaks))

    #-------------------Plot------------------------
    figure=plt.figure(figsize=(12,10),dpi=100)
    ax=figure.add_subplot(111)

    ax.set_xlim([img.shape[1]//2-R, img.shape[1]//2+R])
    ax.set_ylim([img.shape[0]//2-R, img.shape[0]//2+R])
    ax.set_axis_off()
    ax.set_aspect('equal')

    h_used=np.zeros(hough.shape)

    def update(ii,peaks,scale_factor):

        pii=peaks[ii]
        accii,aii,rii=pii

        aidx=np.argmin(abs(aii-h_alpha))
        ridx=np.argmin(abs(rii-h_r))

        if hough[ridx,aidx]<1:
            #continue
            return

        h_used[ridx,aidx]+=1

        distii=(abs(aii-alphas)/scale_alpha)**2+(abs(rii-rs)/scale_r)**2
        idxii=np.argmin(distii)
        idxii=np.unravel_index(idxii,rs.shape)

        p1=[xs[idxii[0]], ys[idxii[0]]]
        p2=[xs[idxii[1]], ys[idxii[1]]]

        p1=[min(p1[0], img.shape[1]-1), min(p1[1],img.shape[0]-1)]
        p2=[min(p2[0], img.shape[1]-1), min(p2[1],img.shape[0]-1)]

        lineidx=getLineByPoints(p1,p2).astype('int')
        intii=img[np.minimum(lineidx[:,1],img.shape[0]-1),\
                np.minimum(lineidx[:,0],img.shape[1]-1)]
        meanintii=np.mean(intii)

        #wii=(float(accii)/maxh)**scale_factor*LINE_WIDTH
        wii=(float(accii)/maxh)**scale_factor*LINE_WIDTH*meanintii
        #wii=float(meanintii)**scale_factor*LINE_WIDTH
        #wii=LINE_WIDTH/3.
        print('peakii = %d, accii = %.1f, maxh = %.1f, wii = %.3f'\
                %(ii,accii,maxh,wii))

        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], '-', color='Gray',
                linewidth=wii,alpha=1.0)



    if ANIMATE:
        #import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation

        anim=FuncAnimation(figure,update,frames=len(peaks),fargs=(peaks,),
                interval=5,
                repeat=True,
                blit=False)
        anim.save('%s_line_animation.mp4' %IMG_FILE,
                fps=100)
    else:

        for ii in range(len(peaks)):
            update(ii,peaks,0.1)

        maxh=hough_masked.max()
        peaks=hlp(accumulator,h_thetas,h_rhos,min_distance=1,min_angle=1,
                threshold=maxh/6.)
        peaks=zip(*peaks)

        print('len(peaks) = %d' %len(peaks))

        for ii in range(len(peaks)):
            update(ii,peaks,4)

        figure.show()


    '''
    for ii,pii in enumerate(peaks):

        accii,aii,rii=pii
        if accii<=maxh/2.:
            continue

        aidx=np.argmin(abs(aii-h_alpha))
        ridx=np.argmin(abs(rii-h_r))

        if hough[ridx,aidx]<1:
            continue

        h_used[ridx,aidx]+=1

        distii=(abs(aii-alphas)/scale_alpha)**2+(abs(rii-rs)/scale_r)**2
        idxii=np.argmin(distii)
        idxii=np.unravel_index(idxii,rs.shape)

        p1=[xs[idxii[0]], ys[idxii[0]]]
        p2=[xs[idxii[1]], ys[idxii[1]]]
        #p1=np.array(p1)
        #p2=np.array(p2)

        wii=(float(accii)/maxh)**2*0.5
        print 'peakii',ii,'accii=',accii,'wii=',wii

        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], '-', color='Gray',
                linewidth=wii,alpha=0.8)
    '''





    '''
    visited=[]
    while nn<n_lines:
        hmax=np.max(hough_masked)
        yidx,xidx=np.where(hough_masked==hmax)
        hough_masked[yidx,xidx]=0

        if np.max(hough_masked)<=maxh/20.:
            break

        coords=zip(yidx,xidx)
        nn+=1
        print 'nn',nn, len(coords), hmax

        for ii in range(len(coords)):
            rii=h_r[coords[ii][0]]
            alphaii=h_alpha[coords[ii][1]]

            distii=(abs(alphaii-alphas)/scale_alpha)**2+(abs(rii-rs)/scale_r)**2
            idxii=np.argmin(distii)
            idxii=np.unravel_index(idxii,rs.shape)

            if idxii in visited:
                continue

            #idxiiy,idxiix=idxii
            #rii2=rs[idxii]
            #alphaii2=alphas[idxii]

            p1=[xs[idxii[0]], ys[idxii[0]]]
            p2=[xs[idxii[1]], ys[idxii[1]]]
            p1=np.array(p1)
            p2=np.array(p2)

            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], '-', color='Gray',
                    linewidth=0.2)

            visited.append(idxii)

            #yhatii=rii2/np.sin(alphaii2)-XX/np.tan(alphaii2)
            #maskii=np.where(abs(YY-yhatii)<=0.1,1,0)

            #rhatii=XX*np.cos(alphaii)+YY*np.sin(alphaii)
            #maskii=np.where((rhatii-rii)<=EPS,1,0)




    '''
