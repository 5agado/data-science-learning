import numpy as np
import argparse
import cv2
import os
import sys
from tqdm import tqdm
import logging
from pathlib import Path
from math import cos, sin, radians

import brush_utils as rb
import image_utils


def rn():
    return np.random.random()


def diff(i1,i2, overblur=False):
    # calculate the difference of 2 float32 BGR images.

    # use rgb
    d = (i1-i2)**2  # * [0.2,1.5,1.3]

    #d = positive_sharpen(np.sum(d, -1), overblur=overblur)
    #d = np.sum(d, -1)
    return d


def limit(x, minimum, maximum):
    return min(max(x, minimum), maximum)


def intrad(orad, fatness):
    #obtain integer radius and shorter-radius
    radius = int(orad)
    srad = int(orad*fatness+1)
    return radius, srad


# get copy of region-of-interest for both src image and canvas
def get_roi(img, canvas, x, y, radius):
    height, width = img.shape[0:2]

    yp = int(min(y + radius, height-1))
    ym = int(max(0, y - radius))
    xp = int(min(x + radius, width-1))
    xm = int(max(0, x - radius))

    # if zero w or h
    if yp<=ym or xp<=xm:
        raise NameError(f'zero ROI at x={x}, y={y} and r={radius}')

    img_roi = img[ym:yp, xm:xp]
    canvas_roi = canvas[ym:yp,xm:xp]

    return img_roi, canvas_roi


# paint one stroke with given config and return the error.
def add_stroke(img, canvas, brush, color, angle, fatness, x, y, r, is_final=False, useoil=False):
    radius, srad = intrad(r, fatness)
    # if brush placement is final, add to canvas and return
    if is_final:
        rb.compose(canvas, brush, x=x, y=y, rad=radius, srad=srad, angle=angle, color=color, useoil=useoil)
        return None
    # otherwise only run on ROI and get error
    else:
        # get ROI areas to calculate error
        ref, before = get_roi(img, canvas, x, y, r)
        after = np.array(before)

        # if useoil here set to true: 2x slow down + instability
        # we disable here, assuming is not influential on error estimation
        rb.compose(after, brush, x=radius, y=radius, rad=radius, srad=srad, angle=angle, color=color, useoil=False)

        err_aftr = np.mean(diff(after, ref))
        return err_aftr


# given err, calculate gradient of parameters wrt to it
def calc_gradient(img, canvas, brush, color, angle, fatness, x, y, radius, gradient_config, err):
    # Color
    b,g,r = color[0],color[1],color[2]
    color_delta = gradient_config['color_delta']
    if color_delta > 0:
        err_aftr = add_stroke(img, canvas, brush, (b+color_delta,g,r), angle, fatness, x, y, radius)
        gb = err_aftr - err

        err_aftr = add_stroke(img, canvas, brush, (b,g+color_delta,r), angle, fatness, x, y, radius)
        gg = err_aftr - err

        err_aftr = add_stroke(img, canvas, brush, (b,g,r+color_delta), angle, fatness, x, y, radius)
        gr = err_aftr - err

        color_gradient = np.array([gb,gg,gr])/color_delta
    else:
        color_gradient = np.array([0.,0.,0.])

    ## Angle
    angle_delta = gradient_config['angle_delta']
    if angle_delta > 0:
        err_aftr = add_stroke(img, canvas, brush, color, (angle+angle_delta)%360, fatness, x, y, radius)
        angle_gradient = (err_aftr - err)/angle_delta
    else:
        angle_gradient = 0.

    ## Position
    pos_delta = gradient_config['pos_delta'] * radius
    if pos_delta > 0:
        err_aftr = add_stroke(img, canvas, brush, color, angle, fatness, x+pos_delta,y, radius)
        x_gradient =  (err_aftr - err)/pos_delta

        err_aftr = add_stroke(img, canvas, brush, color, angle, fatness, x,y+pos_delta, radius)
        y_gradient =  (err_aftr - err)/pos_delta
    else:
        x_gradient = 0.
        y_gradient = 0.

    ## Radius
    radius_delta = gradient_config['radius_delta']
    if radius_delta > 0:
        err_aftr = add_stroke(img, canvas, brush, color, angle, fatness, x,y, radius+radius_delta)
        radius_gradient = (err_aftr - err)/radius_delta
    else:
        radius_gradient = 0.

    return color_gradient, angle_gradient, x_gradient, y_gradient, radius_gradient


def do_descent(color, angle, x, y, radius,
               color_grad, angle_grad, x_grad, y_grad, radius_grad, gradient_config, img):
    color_delta = gradient_config['color_delta']
    color -= (color_grad * .3).clip(max=color_delta, min=-color_delta)
    color = color.clip(max=1., min=0.).astype(np.float32)

    angle_delta = gradient_config['angle_delta']
    angle = (angle - limit(angle_grad * 1000, -angle_delta, angle_delta)) % 360

    radius_delta = gradient_config['radius_delta']
    radius *= (1-limit(radius_grad*20000, -radius_delta, radius_delta))
    radius = limit(radius, 10, 300) # TODO externalize

    pos_delta = gradient_config['pos_delta'] * radius
    height, width = img.shape[0:2]
    x -= limit(x_grad*1000*radius, -pos_delta, pos_delta)
    y -= limit(y_grad*1000*radius, -pos_delta, pos_delta)
    x = limit(x, (radius//2), width-(radius//2))
    y = limit(y, (radius//2), height-(radius//2))

    return color, angle, x, y, radius


def paint_one(img, canvas, distance_map, x, y, angle, radius, magnitude, brush, check_error, useoil, color_neighbor_size,
              gradient_config):
    """
    :param img:
    :param canvas:
    :param x:
    :param y:
    :param angle:
    :param radius:
    :param magnitude:
    :param brush:
    :param check_error:
    :param useoil:
    :param color_neighbor_size:
    :return:
    """
    fatness = 1/(1+magnitude)

    # negate angle because of image coordinates
    dest_x, dest_y =  (int(x + cos(radians(-angle)) * radius), int(y + sin(radians(-angle)) * radius))
    if distance_map is not None:
        min_radius_to_edge = image_utils.get_min_radius_to_edge(distance_map, (x, y), (dest_x, dest_y))

        if min_radius_to_edge > 0:
            radius = min_radius_to_edge

    # set initial color
    color = image_utils.sample_color(img, x=int(x), y=int(y), neighbor_size=color_neighbor_size)

    tryfor = gradient_config['tryfor']
    mintry = gradient_config['mintry']
    for i in range(tryfor):
        # get ROI areas to calculate error
        ref, bef = get_roi(img, canvas, x, y, radius)
        orig_err = np.mean(diff(bef, ref))

        # do the painting
        err = add_stroke(img, canvas, brush, color, angle, fatness, x, y, radius)

        # if error decreased:
        if (not check_error) or (err < orig_err and i >= mintry):
            add_stroke(img, canvas, brush, color, angle, fatness, x, y, radius, is_final=True, useoil=useoil)
            return True, i

        # if not satisfactory, run gradient-descent
        color_grad, angle_grad, x_grad, y_grad, radius_grad = calc_gradient(img, canvas, brush, color, angle, fatness, x, y,
                                                                            radius, gradient_config, err)

        color, angle, x, y, radius = do_descent(color, angle, x, y, radius,
                                                color_grad, angle_grad, x_grad, y_grad, radius_grad, gradient_config, img)

    return False, tryfor


def put_strokes(img, canvas, nb_strokes: int, minrad: int, maxrad: int, brushes: dict,
                salience_img, salience_img_weight: float,
                distance_map,
                sample_map_scale_factor: float, phase_neighbor_size: int,
                color_neighbor_size: int,
                out_dir, iter_idx, check_error: bool, useoil: bool,
                gradient_config: dict,
                border_pct: float):

    def sample_points():
        # sample a lot of points from one error image - save computation cost

        point_list = []
        d = np.sum(diff(canvas, img), -1)
        phase_map, magnitude_map = image_utils.get_phase_and_magnitude(distance_map)
        d = d/d.sum()  # normalize probabilities

        # compose error-map and salience-map
        if salience_img is not None:
            #sample_map = (d * (1.-salience_img_weight)) + (salience_img * salience_img_weight)
            # error map weight by 1-alpha + salience image weight by alpha + combination of full error multiplied by salience image
            sample_map = (d * (1.-salience_img_weight)) + (d * (salience_img * salience_img_weight)) #+ (salience_img * salience_img_weight)
        else:
            sample_map = d

        # set border (based on radius) of salience map to 0
        height, width = img.shape[0:2]
        x_border_size, y_border_size = int(width*border_pct), int(height*border_pct)
        sample_map[:x_border_size, :] = 0.
        sample_map[-x_border_size:, :] = 0.
        sample_map[:, :y_border_size] = 0.
        sample_map[:, -y_border_size:] = 0.

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
        y, x = np.unravel_index(selected_points, sample_map.shape[:2])
        for rx, ry in zip(x, y):

            # as sample map have been scaled down, recompute the indexes for original image
            ry, rx = int(ry / sample_map_scale_factor), int(rx / sample_map_scale_factor)

            angle, magnitude = image_utils.get_point_angle_and_magnitude(rx, ry, phase_map, magnitude_map,
                                                                         phase_neighbor_size=phase_neighbor_size,
                                                                         magnitude_neighbor_size=phase_neighbor_size)
            point_list.append((ry, rx, angle, magnitude))
        return point_list

    def pcasync(point_data):
        y, x, angle, magnitude = point_data

        radius = (rn() * maxrad) + minrad
        brush, key = rb.get_brush(brushes, 'random')
        return paint_one(img, canvas, distance_map, x, y, radius=radius, magnitude=magnitude, brush=brush, angle=angle,
                         check_error=check_error, useoil=useoil, color_neighbor_size=color_neighbor_size,
                         gradient_config=gradient_config) # return num of epoch

    point_list = sample_points()
    res = {}
    for idx, item in enumerate(point_list):
        res[idx] = pcasync(item)
    return res


def load_salience_img(salience_path: Path, salience_img_name: str):
    # if given load salience image
    salience_img = None
    salience_img_proba = None
    if salience_path:
        # get imagepath and extension
        salience_img_ext = None
        for ext in ['.jpg', '.png']:
            if (salience_path / (salience_img_name + ext)).exists():
                salience_img_ext = ext
                break
        if salience_img_ext is None:
            logging.error(f'No salience image for {salience_img_name}')
            #return salience_img
            raise Exception(f'No salience image for {salience_img_name}')

        # load salience image
        salience_img_path = str(salience_path / (salience_img_name + ext))
        salience_img = cv2.imread(salience_img_path, cv2.IMREAD_GRAYSCALE)
        salience_img_proba = salience_img.astype('float32') / 255  # convert to float32
        salience_img_proba = salience_img_proba.clip(0.)
        salience_img_proba = salience_img_proba / salience_img_proba.sum()  # normalizing probability

    return salience_img, salience_img_proba


def paint_images(input_path: Path, output_path: Path, brush_dir: Path, config: dict, nb_epochs: int,
                 salience_path=None, img_postfix=''):
    brushes = rb.load_brushes(brush_dir)

    imgs_paths = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    for img_path in tqdm(imgs_paths):
        logging.info(f'Painting {img_path}')

        # load image
        original_img = cv2.imread(str(img_path))

        # pre-process image
        processed_image = cv2.pyrMeanShiftFiltering(original_img, sp=15, sr=30)
        processed_image = processed_image.astype('float32') / 255  # convert to float32

        # init canvas
        canvas = processed_image.copy()
        canvas[:, :] = 1.5 # initial color as special color (gives MAXINT for diff)

        # load salience img and edges
        salience_img, salience_img_proba = load_salience_img(salience_path, img_path.stem)
        edges, uint_edges = image_utils.get_edges(processed_image, min_hyst_val=config['edge_min_hyst'],
                                                  max_hyst_val=config['edge_max_hyst'])
        distance_map = image_utils.get_distance_map(uint_edges)
        if salience_img_proba is not None:
            salience_plus_edges = salience_img_proba * edges
            salience_plus_edges /= salience_plus_edges.sum()  # normalizing probability
        else:
            salience_plus_edges = edges / edges.sum()

        # log some data
        if output_path:
            img_out_path = output_path / (img_path.stem + img_postfix)
            img_out_path.mkdir(exist_ok=True, parents=True)

            # write config
            with open(str(img_out_path / 'config.txt'), 'w+') as f:
                f.write(str(config) + '\n')

            cv2.imwrite(str(img_out_path / 'original.png'), original_img)
            cv2.imwrite(str(img_out_path / 'distance.png'), distance_map * 255)
            cv2.imwrite(str(img_out_path / 'pre_process.png'), processed_image * 255)
            if salience_img is not None:
                cv2.imwrite(str(img_out_path / 'salience.png'), salience_img)
            cv2.imwrite(str(img_out_path / 'edges.png'), uint_edges)

        # set brush radius
        minrad = max(canvas.shape[:2]) * config['min_radius_to_image_factor']
        maxrad = max(canvas.shape[:2]) * config['max_radius_to_image_factor']

        max_radii = np.linspace(minrad, maxrad, nb_epochs, dtype=int)[::-1] if nb_epochs > 1 else [maxrad]
        min_radii = [r - r//config['radius_diff_factor'] for r in max_radii]

        nb_strokes = config['nb_strokes']  # number of stroke tries per epoch

        # paint
        for i in tqdm(range(nb_epochs)):
            succeeded = 0  # how many strokes being placed
            avg_step = 0.  # average step of gradient descent performed

            # paint strokes
            res = put_strokes(processed_image, canvas, nb_strokes[i], min_radii[i], max_radii[i],
                              brushes=brushes,
                              salience_img=salience_img_proba if not config['refine_edges'][i] else salience_plus_edges,
                              salience_img_weight=config['salience_img_weights'][i],
                              distance_map=distance_map,
                              sample_map_scale_factor=config['sample_map_scale_factor'],
                              phase_neighbor_size=int(config['phase_neighbor_size'][i]),
                              color_neighbor_size=int(config['color_neighbor_size'][i]),
                              out_dir=output_path / img_path.stem, iter_idx=i,
                              check_error=config['check_error'][i],
                              useoil=config['use_oil'],
                              gradient_config=config['gradient_config'],
                              border_pct=config['border_pct'])

            # some running stats
            for r in res:
                status, step = res[r]
                avg_step += step
                succeeded += 1 if status else 0
            avg_step /= nb_strokes[i]
            logging.debug(f'succeeded: {succeeded}, avg step: {avg_step}')

            # save progress
            if output_path:
                cv2.imwrite(str(img_out_path / f'{i:04d}.png'), canvas * 255)


def main(_=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Assisted Painting')

    parser.add_argument('-i', '--input-path', required=True)
    parser.add_argument('-o', '--output-path', required=True)
    parser.add_argument('-e', '--nb-epochs', type=int, default=10)
    parser.add_argument('-b', '--brush-dir', default=Path(os.path.dirname(__file__)) / 'brushes')
    parser.add_argument('--salience_path', default=None)
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    brush_dir = Path(args.brush_dir)
    salience_path = None if args.salience_path is None else Path(args.salience_path)
    nb_epochs = args.nb_epochs
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main_config = {
        'max_radius_to_image_factor': 1/9,
        'min_radius_to_image_factor': 1/50,
        # for salience
        #'salience_img_weights': [0.] * 2 + list(np.linspace(0.2, 1.0, 6)) + [1.] * nb_epochs,
        # for depth
        'salience_img_weights': list(np.linspace(0.2, 0.8, 7)) + [0.9] * nb_epochs,
        'check_error': [False] * 2 + [True] * nb_epochs,
        'gd_tryfor': 7,
        'gd_mintry': 1,
        'nb_strokes': [150] * 3 + [300] * 4 + [390] * nb_epochs,
        'radius_diff_factor': 20,
        'sample_map_scale_factor': 0.3,
        'use_oil': True,
        'phase_neighbor_size': [2] * nb_epochs, #np.linspace(2, 7, num=nb_epochs)[::-1],
        'color_neighbor_size': [40] * 2 + [18] * 2 + [4] * nb_epochs,
        'refine_edges': [False] * (nb_epochs - 1) + [False] * nb_epochs,
        'edge_min_hyst': 80,
        'edge_max_hyst': 150,
        'border_pct': 0.05,
        'gradient_config': {
            'color_delta': 0, #1e-4,
            'angle_delta': 5,
            'pos_delta': 0.2,   # proportional to radius
            'radius_delta': 0.2,
            'tryfor': 3,
            'mintry': 1,
        }
    }


    details_config = {
        'max_radius_to_image_factor': 1/20,
        'min_radius_to_image_factor': 1/90,
        'salience_img_weights': list(np.linspace(0.9, 1., 3)) + [1.] * nb_epochs,
        'check_error': [True] * 2 + [True] * nb_epochs,
        'nb_strokes': [50] * 2 + [150] * 2 + [200] * 2 + [250] * 2,
        'radius_diff_factor': 20,
        'sample_map_scale_factor': 0.3,
        'use_oil': True,
        'phase_neighbor_size': [2] * nb_epochs,
        'color_neighbor_size': [40] * 2 + [18] * 2 + [4] * nb_epochs,
        'refine_edges': [False] * nb_epochs,
        'edge_min_hyst': 80,
        'edge_max_hyst': 150,
        'border_pct': 0.05,
        'gradient_config': {
            'color_delta': 0,
            'angle_delta': 5,
            'pos_delta': 0.2,   # proportional to radius
            'radius_delta': 0.2,
            'tryfor': 5,
            'mintry': 1,
        }
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
        'color_neighbor_size': [5] * nb_epochs,
        'refine_edges': [False] * nb_epochs,
        'edge_min_hyst': 100,
        'edge_max_hyst': 300,
        'border_pct': 0.0,
        'gradient_config': {
            'color_delta': 0,
            'angle_delta': 0,
            'pos_delta': 0,
            'radius_delta': 0,
            'tryfor': 0,
            'mintry': 0,
        }
    }

    paint_images(input_path, output_path, brush_dir, main_config, nb_epochs, salience_path)
    return

    config_gs = {
        #'sample_map_scale_factor': np.linspace(0.1, 1.5, 5)
        #'max_radius_to_image_factor': [1/v for v in np.linspace(5, 50, 10)]
        #'min_radius_to_image_factor': [1/v for v in np.linspace(50, 200, 10)]
        #'check_error': [[False] * nb_epochs, [True] * nb_epochs],
        #'phase_neighbor_size': [[val] * nb_epochs for val in np.linspace(1, 15, 10)],
        #'nb_strokes': [[val] * nb_epochs for val in np.arange(50, 401, 70)]
        #'gd_tryfor': np.arange(1, 32, 6)
    }
    for param, values in config_gs.items():
        for i, val in enumerate(values):
            main_config[param] = val
            paint_images(input_path, output_path, brush_dir, main_config, nb_epochs, salience_path,
                         img_postfix= f'_{param}_{i}')


if __name__ == "__main__":
    main(sys.argv[1:])

