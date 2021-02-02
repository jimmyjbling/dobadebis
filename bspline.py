import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('pdf')

from geomdl import BSpline, knotvector, fitting, exchange

json_file = "C:/Users/welln/OneDrive/Research/Li_Lab/OPENCV/3_CHEMBL158617_bspline.json"

curve = BSpline.Curve()
curve = exchange.import_json(json_file)[0]
curve.delta = 0.001

curve_ctrls_list = curve.ctrlpts
curve_ctrls_array = np.array(curve_ctrls_list)

evalpts = np.array(curve.evalpts)
plt.plot(evalpts[:,0], evalpts[:,1], ':k', lw=0.5)
plt.axis('off')
plt.axis('scaled')
plt.savefig('tmp.png', dpi=300, bbox_inches='tight')
