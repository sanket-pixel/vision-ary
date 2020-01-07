import numpy as np
import cv2
import matplotlib.pyplot as plt



def display_image(image,title="random"):
    cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()


# ===========================================
# RUNNING
# ===========================================



def sobel_gradient(image):
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image,-1,np.fliplr(np.flipud(sobel)))
def gradient(image):
    grad = np.gradient(image)
    return np.hypot(grad[1], grad[0])
def phi_x(phi):
    return np.gradient(phi)[1]
def phi_y(phi):
    return np.gradient(phi)[0]
def phi_xx(phi):
    filter_xx = np.array([1,-2,1]).reshape(1,-1)
    return cv2.filter2D(phi,-1,filter_xx)
def phi_yy(phi):
    filter_yy = np.array([1, -2, 1])
    return cv2.filter2D(phi, -1, filter_yy)
def phi_xy(phi):
    filter_xy = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
    return 0.25*cv2.filter2D(phi,-1,filter_xy)

def metric(gradient):
    return 1/(1+np.abs(gradient))

def mean_curvature_motion(phi,eps):
    numerator = phi_xx(phi)*(phi_y(phi)**2) - 2*phi_x(phi)*phi_y(phi)*phi_xy(phi) + phi_yy(phi)*(phi_x(phi)**2)
    denominator = phi_x(phi)**2 + phi_y(phi)**2 + eps
    return numerator/denominator

def front_propogation(phi,w):
    wx = np.gradient(w)[1]
    wy = np.gradient(w)[0]
    wx_max = np.max((wx,np.zeros_like(wx)),axis=0)
    diffx_forward= np.pad(np.diff(phi, axis=1), ((0, 0), (0, 1)))
    wx_min =  np.min((wx,np.zeros_like(wx)),axis=0)
    diffx_backward = np.pad(np.diff(phi, axis=1), ((0, 0), (1, 0)))
    wy_max = np.max((wy,np.zeros_like(wy)),axis=0)
    diffy_forward = np.pad(np.diff(phi, axis=0), ((0, 1), (0, 0)))
    wy_min = np.min((wy,np.zeros_like(wy)),axis=0)
    diffy_backward = np.pad(np.diff(phi, axis=0), ((1, 0), (0, 0)))
    front_prop_term = wx_max*diffx_forward + wx_min*diffx_backward + wy_max*diffy_forward + wy_min*diffy_backward
    return front_prop_term
if __name__ == '__main__':
    plt.pause(4)
    n_steps = 6000
    plot_every_n_step = 100

    Im, phi = load_data()
    image_gradient = gradient(Im)
    w = metric(image_gradient)
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here
    # ------------------------
    eta = 1/(4*np.max(w))
    eps = 0.0001
    for t in range(n_steps):
        phi_old = np.copy(phi)
        term_1 = w*mean_curvature_motion(phi,eps)
        term_2 = front_propogation(phi,w)
        phi = phi + eta*(term_1+term_2)
        if t % plot_every_n_step == 0:
            print(t)
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))
            contour = get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=10)
            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.0001)
    plt.show()
