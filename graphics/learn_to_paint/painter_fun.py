import numpy as np
import argparse
import cv2
import sys
import math
from tqdm import tqdm
import logging
from pathlib import Path
import os

import brush_utils as rb
import image_utils


def rn():
    return np.random.random()


def positive_sharpen(i, overblur=False, coeff=8.): #no darken to original image
    # emphasize the edges
    blurred = cv2.blur(i,(5,5))
    sharpened = i + (i - blurred) * coeff
    if overblur:
        return cv2.blur(np.maximum(sharpened,i),(11,11))
    return cv2.blur(np.maximum(sharpened,i),(3,3))


def diff(i1,i2,overblur=False):
    # calculate the difference of 2 float32 BGR images.

    # use rgb
    d = (i1-i2)# * [0.2,1.5,1.3]
    d = d*d

    #d = positive_sharpen(np.sum(d, -1), overblur=overblur)
    d = np.sum(d, -1)
    return d


def wherediff(i1=None, i2=None):
    # find out where max difference point is.
    d = diff(i1, i2, overblur=True)

    i, j = np.unravel_index(d.argmax(), d.shape)
    return i, j, d


def limit(x, minimum, maximum):
    return min(max(x, minimum), maximum)


def paint_one(img, canvas, x, y, angle, oradius, magnitude, brush, check_error=True, useoil=False):
    fatness = 1/(1+magnitude*6)

    def intrad(orad):
        #obtain integer radius and shorter-radius
        radius = int(orad)
        srad = int(orad*fatness+1)
        return radius, srad

    # set initial color
    # sample color from image => converges faster.
    c = img[int(y), int(x), :]

    delta = 1e-4

    # get copy of square ROI area, to do drawing and calculate error.
    def get_roi(newx, newy, newrad):
        radius, srad = intrad(newrad)

        height, width = img.shape[0:2]

        yp = int(min(newy+radius, height-1))
        ym = int(max(0, newy-radius))
        xp = int(min(newx+radius, width-1))
        xm = int(max(0, newx-radius))

        # if zero w or h
        if yp<=ym or xp<=xm:
            raise NameError('zero roi')

        ref = img[ym:yp, xm:xp]
        before = canvas[ym:yp,xm:xp]
        after = np.array(before)

        return ref, before, after

    # paint one stroke with given config and return the error.
    def paint_aftr_w(color, angle, nx, ny, nr, useoil=False):
        ref, before, after = get_roi(nx, ny, nr)
        radius, srad = intrad(nr)

        rb.compose(after, brush, x=radius, y=radius, rad=radius, srad=srad, angle=angle, color=color,
                   usefloat=True, useoil=useoil)
        # if useoil here set to true: 2x slow down + instability

        err_aftr = np.mean(diff(after, ref))
        return err_aftr

    # finally paint the same stroke onto the canvas.
    def paint_final_w(color, angle, nr, useoil=False):
        radius, srad = intrad(nr)

        rb.compose(canvas, brush, x=x, y=y, rad=radius, srad=srad, angle=angle, color=color, usefloat=True, useoil=useoil)
        # enable oil effects on final paint.

        # np.float64 will cause problems
        #rec = [x,y,radius,srad,angle,color[0],color[1],color[2],brushname]
        #rec = [float(r) if type(r)==np.float64 or type(r)==np.float32 else r for r in rec]
        #record(rec)
        # log it!

    # given err, calculate gradient of parameters wrt to it
    def calc_gradient(err):
        b,g,r = c[0],c[1],c[2]
        cc = b,g,r

        err_aftr = paint_aftr_w((b+delta,g,r),angle,x,y,oradius)
        gb = err_aftr - err

        err_aftr = paint_aftr_w((b,g+delta,r),angle,x,y,oradius)
        gg = err_aftr - err

        err_aftr = paint_aftr_w((b,g,r+delta),angle,x,y,oradius)
        gr = err_aftr - err

        err_aftr = paint_aftr_w(cc,(angle+5.)%360,x,y,oradius)
        ga = err_aftr - err

        #err_aftr = paint_aftr_w(cc,angle,x+2,y,oradius)
        #gx =  err_aftr - err
        gx = 0

        #err_aftr = paint_aftr_w(cc,angle,x,y+2,oradius)
        #gy =  err_aftr - err
        gy = 0

        err_aftr = paint_aftr_w(cc,angle,x,y,oradius+3)
        gradius = err_aftr - err

        return np.array([gb,gg,gr])/delta,ga/5,gx/2,gy/2,gradius/3,err

    # max and min steps for gradient descent
    tryfor = 12
    mintry = 3

    for i in range(tryfor):
        try: # might have error
            # what is the error at ROI?
            ref, bef, aftr = get_roi(x, y, oradius)
            orig_err = np.mean(diff(bef, ref))

            # do the painting
            err = paint_aftr_w(c, angle, x, y, oradius)

            # if error decreased:
            if (not check_error) or (err < orig_err and i >= mintry):
                paint_final_w(c, angle, oradius, useoil=useoil)
                return True, i

            # if not satisfactory
            # calculate gradient
            grad, anglegrad, gx, gy, gradius, err = calc_gradient(err)

        except NameError as e:
            print(e)
            print('error within calc_gradient')
            return False, i

        #if printgrad: #debug purpose.
        #    if i==0:
        #        print('----------')
        #        print('orig_err',orig_err)
        #    print('ep:{}, err:{:3f}, color:{}, angle:{:2f}, xy:{:2f},{:2f}, radius:{:2f}'.format(i,err,c,angle,x,y,oradius))

        # do descend
        if i < tryfor-1:
            c = c - (grad*.3).clip(max=0.3,min=-0.3)
            c = c.clip(max=1.,min=0.)
            angle = (angle - limit(anglegrad*100000,-5,5))%360
            #x = x - limit(gx*1000*radius,-3,3)
            #y = y - limit(gy*1000*radius,-3,3)
            oradius = oradius* (1-limit(gradius*20000,-0.2,.2))
            oradius = limit(oradius,7,100)

            # print('after desc:x:{:2f},y:{:2f},angle:{:2f},oradius:{:5f}'
            # .format(x,y,angle,oradius))

    return False, tryfor


def put_strokes(img, canvas, nb_strokes: int, minrad: int, maxrad: int, brushes: dict,
                salience_img, salience_img_weight: float,
                sample_map_scale_factor: float, phase_neighbor_size: int,
                out_dir, iter_idx, check_error=True, useoil=False):
    logging.debug(f'minrad {minrad}')
    logging.debug(f'maxrad {maxrad}')

    def sample_points():
        # sample a lot of points from one error image - save computation cost

        point_list = []
        y, x, d = wherediff(canvas, img)
        phase_map, magnitude_map = image_utils.get_phase_and_magnitude(img)
        d = d/d.sum()  # normalize probabilities

        # compose error-map and salience-map
        if salience_img is not None:
            sample_map = (d * (1.-salience_img_weight)) + (salience_img * salience_img_weight)
        else:
            sample_map = d

        # scale down map
        sample_map = cv2.resize(sample_map, None, fx=sample_map_scale_factor, fy=sample_map_scale_factor,
                                interpolation=cv2.INTER_AREA)

        sample_map = sample_map / sample_map.sum()  # normalize probabilities

        if out_dir:
            cv2.imwrite(str(out_dir / f'sample_map_{iter_idx:04d}.png'),
                        cv2.normalize(sample_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

        # select point by given probability map
        sample_map_flat = sample_map.ravel()
        sample_map_range = range(sample_map_flat.size)
        selected_points = np.random.choice(sample_map_range, size=nb_strokes, p=sample_map_flat)

        # iterate through selected points
        for yx in selected_points:
            ry, rx = np.unravel_index(yx, sample_map.shape[:2])

            # as sample map have been scaled down, recompute the indexes for original image
            ry, rx = int(ry / sample_map_scale_factor), int(rx / sample_map_scale_factor)

            # get gradient orientation info from phase map (phase should be between [0,2pi))
            # compute an average phase around the point, for an area proportional to brush size
            phase = phase_map[max(0, ry-phase_neighbor_size):ry+phase_neighbor_size,
                              max(0, rx-phase_neighbor_size):rx+phase_neighbor_size].mean()

            # choose direction perpendicular to gradient
            angle = (((phase/math.pi)*180)+90) % 360

            magnitude = magnitude_map[max(0, ry-phase_neighbor_size):ry+phase_neighbor_size,
                                      max(0, rx-phase_neighbor_size):rx+phase_neighbor_size].mean()

            point_list.append((ry, rx, angle, rn()*rn()))
        return point_list

    def pcasync(point_data):
        y, x, angle, magnitude = point_data

        radius = (rn() * maxrad) + minrad
        brush, key = rb.get_brush(brushes, 'random')
        return paint_one(img, canvas, x, y, oradius=radius, magnitude=magnitude, brush=brush, angle=angle,
                         check_error=check_error, useoil=useoil) # retun num of epoch

    point_list = sample_points()
    res = {}
    for idx, item in enumerate(point_list):
        res[idx] = pcasync(item)
        #cv2.imwrite(str(out_dir) + f'{idx:04d}.png', canvas * 255)
    return res


def load_salience_img(salience_path: Path, salience_img_name: str, shape, test_border=False):
    # if given load salience image
    salience_img = None
    if salience_path:
        # get imagepath and extension
        salience_img_ext = None
        for ext in ['.jpg', '.png', '.npy']:
            if (salience_path / (salience_img_name + ext)).exists():
                salience_img_ext = ext
                break
        if salience_img_ext is None:
            raise Exception(f'No salience image for {salience_img_name}')
        salience_img_path = str(salience_path / (salience_img_name + ext))

        # if numpy file
        if salience_img_ext == '.npy':
            with open(salience_img_path) as f:
                salience_img = np.load(salience_img_path)
            salience_img = cv2.resize(salience_img, shape)
            cv2.imwrite(str(salience_path / f'salience_{salience_img_name}.png'), salience_img * 255)
        else:
            salience_img = cv2.imread(salience_img_path, cv2.IMREAD_GRAYSCALE)
            salience_img = salience_img.astype('float32') / 255  # convert to float32
        salience_img = salience_img.clip(0.)
        salience_img = salience_img / salience_img.sum()  # normalizing probability

    # secondary experiments with fixed mask
    if test_border:
        mask = np.ones((shape[1], shape[0]), np.uint8) * 255
        cv2.rectangle(mask, (120, 100), (shape[0]-120, shape[1]-60), (0, 0, 0), -1, cv2.LINE_AA)
        #cv2.oval(mask, (shape[1]//2, shape[0]//2), int(shape[1]/2), (0, 0, 0), -1, cv2.LINE_AA)
        mask = cv2.blur(mask, (201, 201))

        salience_img = mask.astype('float32') / 255  # convert to float32
        salience_img = salience_img / salience_img.sum()  # normalizing probability

    return salience_img


def paint_images(input_path: Path, output_path: Path, brush_dir: Path, config: dict, nb_epochs: int, salience_path=None):
    brushes = rb.load_brushes(brush_dir)

    imgs_paths = list(input_path.glob('*.png'))
    imgs_paths.extend(list(input_path.glob('*.jpg')))
    for img_path in tqdm(imgs_paths):
        logging.info(f'Painting {img_path}')
        if output_path:
            img_out_path = output_path / img_path.stem
            img_out_path.mkdir(exist_ok=True, parents=True)

            # write config
            with open(str(img_out_path / 'config.txt'), 'w+') as f:
                f.write(str(config) + '\n')

        # load image
        img = cv2.imread(str(img_path))

        test_border = config.get('test_border', False)
        if test_border:
            border_size = 100
            border_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255]*3)
            border_img = border_img.astype('float32') / 255  # convert to float32
            img = cv2.copyMakeBorder(img, border_size//2, border_size//2, border_size//2, border_size//2, cv2.BORDER_CONSTANT, value=[255]*3)
            img = cv2.resize(img, border_img.shape[:2][::-1])

        img = img.astype('float32') / 255  # convert to float32
        if output_path:
            cv2.imwrite(str(img_out_path / 'original.png'), img * 255)

        # load salience img
        salience_img = load_salience_img(salience_path, img_path.stem, img.shape[:2][::-1],  test_border=test_border)

        # init canvas
        if test_border:
            canvas = border_img.copy()
        else:
            canvas = img.copy()
            canvas[:, :] = 1.

        # set brush radius
        minrad = max(img.shape[:2]) * config['min_radius_to_image_factor']
        maxrad = max(img.shape[:2]) * config['max_radius_to_image_factor']

        max_radii = np.linspace(minrad, maxrad, nb_epochs, dtype=int)[::-1] if nb_epochs > 1 else [maxrad]
        min_radii = [r - r//config['radius_diff_factor'] for r in max_radii]

        nb_strokes = config['nb_strokes']
        # number of stroke tries per batch, sent to thread pool
        # smaller number decreases efficiency

        for i in tqdm(range(nb_epochs)):
            succeeded = 0  # how many strokes being placed
            avg_step = 0.  # average step of gradient descent performed

            # apply morphology open to smooth the outline
            #kernel_size = max(2, (10 // (i + 1)))
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            #blurred_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            kernel_size = max(50, (150 // (i + 1)))
            blurred_img = cv2.bilateralFilter(img, 9, kernel_size, kernel_size)

            # save progress
            #if output_path:
            #    cv2.imwrite(str(img_out_path / f'blurred_{i:04d}.png'), blurred_img * 255)

            # paint strokes
            res = put_strokes(blurred_img, canvas, nb_strokes[i], min_radii[i], max_radii[i],
                              brushes=brushes,
                              salience_img=salience_img, salience_img_weight=config['salience_img_weights'][i],
                              sample_map_scale_factor=config['sample_map_scale_factor'],
                              phase_neighbor_size=int(config['phase_neighbor_size'][i]),
                              out_dir=output_path / img_path.stem, iter_idx=i, # img_out_path / f'{i:04d}')
                              check_error=config['check_error'][i],
                              useoil=config['use_oil'])

            # some running stats
            for r in res:
                status, step = res[r]
                avg_step += step
                succeeded += 1 if status else 0
            avg_step /= nb_strokes[i]
            logging.info(f'succeeded: {succeeded}, avg step: {avg_step}')

            # save progress
            if output_path:
                cv2.imwrite(str(img_out_path / f'{i:04d}.png'), canvas * 255)


def main(_=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Assisted Painting')

    parser.add_argument('-i', '--input-path', required=True)
    parser.add_argument('-o', '--output-path', required=True)
    parser.add_argument('-e', '--nb-epochs', type=int, default=10)
    parser.add_argument('-c', '--config-path', default='')
    parser.add_argument('-b', '--brush-dir', default=Path(os.path.dirname(__file__)) / 'brushes')
    parser.add_argument('--salience_path', default=None)
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    config_path = Path(args.config_path)
    brush_dir = Path(args.brush_dir)
    salience_path = None if args.salience_path is None else Path(args.salience_path)
    nb_epochs = args.nb_epochs
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main_config = {
        'max_radius_to_image_factor': 1/10,
        'min_radius_to_image_factor': 1 / 80,  # option of (50 + nb_epochs)
        'salience_img_weights': [0.] * 2 + list(np.linspace(0.3, 1.0, nb_epochs-2)) + [1.] * nb_epochs,
        'check_error': [False] * 2 + [True] * nb_epochs,
        'nb_strokes': [100] * 2 + [256] * 3 + [300] * 4 + [300] * nb_epochs,
        'radius_diff_factor': 20,
        'sample_map_scale_factor': 0.5,
        'use_oil': True,
        'phase_neighbor_size': np.linspace(5, 10, num=nb_epochs)[::-1],
    }

    border_config = {
        'max_radius_to_image_factor': 1/20,
        'min_radius_to_image_factor': 1/60,  # option of (50 + nb_epochs)
        'salience_img_weights': [1.] * nb_epochs,
        'check_error': [False] * nb_epochs,
        'nb_strokes': [360] * nb_epochs,
        'radius_diff_factor': 20,
        'sample_map_scale_factor': 0.5,
        'use_oil': True,
        'phase_neighbor_size': np.linspace(5, 10, num=nb_epochs)[::-1],
        'test_border': True,
    }

    test_config = {
        'max_radius_to_image_factor': 1 / 100,
        'min_radius_to_image_factor': 1 / 100,
        'salience_img_weights': [1.] * nb_epochs,
        'check_error': [False] * nb_epochs,
        'nb_strokes': [300] * nb_epochs,
        'radius_diff_factor': 1,
        'sample_map_scale_factor': 1.,
        'use_oil': False,
        'phase_neighbor_size': [5] * nb_epochs,
    }

    paint_images(input_path, output_path, brush_dir, main_config, nb_epochs, salience_path)


if __name__ == "__main__":
    main(sys.argv[1:])

