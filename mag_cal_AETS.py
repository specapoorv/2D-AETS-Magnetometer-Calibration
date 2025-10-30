import numpy as np
import matplotlib.pyplot as plt

def get_data(csv_path):
    data = np.loadtxt(csv_path, delimiter=',', dtype=float, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def fit_ellipse(x, y):
    #constructing design matrix D, x and y are 1*N matrix
    x = np.array(x, dtype=float) #all are 1d arrays of shape (1165, )
    y = np.array(y, dtype=float)
    xy = np.multiply(x,y)
    x_sq = np.multiply(x,x)
    y_sq = np.multiply(y, y)
    one = np.ones(np.shape(x))

    D = np.vstack((x_sq, y_sq, xy, x, y, one)) #shape (1165, 6)
    D = D.T

    # H is just D without y_sq

    _, _, Vt = np.linalg.svd(D) # i dont understand the math of svd and all 
    p = Vt[-1, :]

    a, b, c, d, e, f = p
    return a, b, c, d, e, f

def get_parameters(a, b, c, d, e, f, B_h):

    x1 = a/c
    x2 = b/c
    x3 = d/c
    x4 = e/c
    x5 = f/c

    if x1 < 0:
        x1, x2, x3, x4, x5 = -x1, -x2, -x3, -x4, -x5

    #X is matrix of all these x1 , x2 terms and we can write Y = HX

    bx = -1 * (2*x3 - x2*x4) / (-x2*x2 + 4*x1)
    by = -1 * (2*x1*x4 - x2*x3) / (-x2*x2 + 4*x1)

    # rho = np.arcsin(x2 / 2 * (np.sqrt(x1)))
    # calculating like this was giving some error so did this instead
    rho = 0.5 * np.arctan2(b, a - c)


    # Common numerator
    numerator = 2 * np.sqrt(x1*(x5 * x2**2) - (x2 * x3 * x4 )+ (x2**3) + (x1 * x2 * x4**2) - 4 * (x1 * x5))

    # Denominators
    denominator_x = B_h*np.sqrt(x1) *(-x2**2 + 4 * x1)
    denominator_y = B_h*(-x2**2 + 4 * x1)

    s_x = np.abs(numerator/denominator_x)
    s_y = np.abs(numerator/denominator_y)

    #B_h is local horizontal magnetic field we can take the local field from a geomagnetic model
    #right now i am taking it to be 40 micro tesla for IITM, now taking it 3.0 idk why

    return bx, by, rho, s_x, s_y

def calibrate(x, y, bx, by, rho, sx, sy):
    rho = 1.57 - rho
    R = np.array([[np.cos(-rho), -np.sin(-rho)],
              [np.sin(-rho),  np.cos(-rho)]])
    
    data_raw = np.vstack((x, y)) + np.array([[bx], [by]])
    scale_mat = np.diag([1/sx, 1/sy])
    data_corr = scale_mat @ R @ data_raw

    x_corr, y_corr = data_corr[0], data_corr[1]

    return x_corr, y_corr


def plotter(x, y, x_corr, y_corr):
    """
    Plot raw (x, y) and corrected (x_corr, y_corr) magnetometer data.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='r', s=5, label='Raw data')
    plt.scatter(x_corr, y_corr, color='g', s=5, label='Corrected data')
    plt.axis('equal')
    plt.legend()
    plt.title('Magnetometer Calibration: Raw vs Corrected')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def main():
    x, y = get_data("/home/specapoorv/magnetometer_calibration/mag_data.csv")
    #x and y is vector
    a, b, c, d, e, f = fit_ellipse(x, y)
    norm_factor = np.sqrt(a**2 + b**2 + c**2)
    a, b, c, d, e, f = a/norm_factor, b/norm_factor, c/norm_factor, d/norm_factor, e/norm_factor, f/norm_factor
    b_x, b_y, rho, s_x, s_y = get_parameters(a, b, c, d, e, f, B_h=2.0)
    print(f"offsets = ({b_x}, {b_y})")
    print(f"rho = {rho}")
    print(f"scaling factors = {s_x}, {s_y}")
    x_corr, y_corr = calibrate(x, y, b_x, b_y, rho, s_x, s_y)
    plotter(x, y, x_corr, y_corr)


if __name__ == "__main__":
    main()


