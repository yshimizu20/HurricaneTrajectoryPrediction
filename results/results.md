<h1>Testing (sensitivity, solver) pairs</h1>

| Experiment Type             |          one_run           |              all             |         next_only         |
|-----------------------------|:--------------------------:|:----------------------------:|:-------------------------:|
| RungeKutta4, autograd       |      (262.089, 50)         |        (539.792, 50)        |       (62.882, 40)       |
| RungeKutta4, adjoint        |      (301.695, 15)         |        (637.713, 15)        |       (46.683, 15)       |
| DormandPrince45, adjoint    |      (335.383, 35)         |       (1030.119, 65)        |       (71.260, 15)       |
| Tsitouras45, adjoint        |     (384.867, 115)         |        (709.616, 150)       |       (87.235, 25)       |
| Tsitouras45, autograd       |     (388.782, 365)         |        (645.367, 250)       |      (77.957, 400)       |
| midpoint, autograd          |     (409.978, 385)         |        (701.146, 155)       |       (87.806, 55)       |
| euler, adjoint              |     (450.258, 340)         |        (648.686, 115)       |      (91.683, 390)       |
| euler, autograd             |      (452.290, 90)         |        (722.552, 55)        |      (104.315, 90)       |
| midpoint, adjoint           |      (511.959, 70)         |       (1049.561, 150)       |      (112.104, 400)      |
| DormandPrince45, autograd   |      (530.311, 60)         |       (1003.788, 180)       |      (100.513, 95)       |


<h1>Testing (sensitivity, forward_solver, backward_solver) triplets</h1>

| Experiment Type                            |          one_run           |             all            |        next_only        |
|--------------------------------------------|:--------------------------:|:--------------------------:|:-----------------------:|
| euler, midpoint, adjoint                   |      (208.125, 15)         |      (433.159, 15)         |      (43.576, 15)       |
| euler, DormandPrince45, adjoint            |      (217.403, 30)         |      (446.770, 10)         |      (43.889, 15)       |
| DormandPrince45, Tsitouras45, autograd    |     (227.101, 135)         |      (449.525, 145)        |      (30.216, 15)       |
| DormandPrince45, midpoint, autograd        |     (237.956, 40)          |      (468.279, 100)        |      (30.163, 15)       |
| midpoint, euler, autograd                  |     (239.201, 125)         |      (560.949, 125)        |      (55.095, 145)      |
| Tsitouras45, midpoint, autograd            |     (240.536, 20)          |      (564.485, 20)         |      (47.618, 20)       |
| midpoint, DormandPrince45, autograd        |     (241.217, 255)         |      (534.821, 425)        |      (54.682, 295)      |
| Tsitouras45, midpoint, adjoint             |     (278.311, 160)         |      (591.599, 105)        |      (46.138, 385)      |
| euler, RungeKutta4, adjoint                |     (282.426, 45)          |      (522.991, 30)         |      (64.216, 25)       |
| euler, Tsitouras45, adjoint                |     (283.864, 50)          |      (527.569, 95)         |      (64.611, 20)       |
| Tsitouras45, euler, autograd               |     (292.461, 100)         |      (1018.329, 0)         |      (50.963, 395)      |
| midpoint, RungeKutta4, autograd            |     (294.439, 140)         |      (612.877, 140)        |      (70.961, 30)       |
| RungeKutta4, DormandPrince45, adjoint      |     (298.938, 185)         |      (573.885, 185)        |      (67.658, 205)      |
| DormandPrince45, euler, adjoint            |     (300.479, 345)         |      (519.205, 125)        |      (70.494, 125)      |
| midpoint, Tsitouras45, adjoint             |     (302.756, 10)          |      (599.886, 10)         |      (54.548, 35)       |
| midpoint, Tsitouras45, autograd            |     (304.086, 125)         |      (620.431, 125)        |      (71.064, 335)      |
| DormandPrince45, RungeKutta4, adjoint      |     (320.094, 775)         |      (1005.947, 385)       |      (63.742, 70)       |
| Tsitouras45, euler, adjoint                |     (325.090, 35)          |      (454.638, 15)         |      (62.315, 400)      |
| midpoint, euler, adjoint                   |     (325.482, 375)         |      (709.284, 25)         |      (56.251, 375)      |
| RungeKutta4, euler, adjoint                |     (335.324, 65)          |      (594.071, 155)        |      (70.476, 10)       |
| RungeKutta4, euler, autograd               |     (338.924, 30)          |      (1005.365, 125)       |      (73.173, 30)       |
| RungeKutta4, midpoint, adjoint             |     (341.481, 400)         |      (642.492, 375)        |      (61.257, 345)      |
| midpoint, RungeKutta4, adjoint             |     (344.473, 65)          |      (626.296, 15)         |      (54.846, 35)       |
| RungeKutta4, Tsitouras45, adjoint          |     (355.732, 400)         |      (648.365, 400)        |      (61.611, 215)      |
| midpoint, DormandPrince45, adjoint         |     (357.118, 30)          |      (736.008, 20)         |      (56.654, 30)       |
| RungeKutta4, DormandPrince45, autograd     |     (366.773, 10)          |      (1000.101, 125)       |      (73.233, 20)       |
| RungeKutta4, Tsitouras45, autograd         |     (366.825, 400)         |      (492.146, 245)        |      (80.863, 65)       |
| RungeKutta4, midpoint, autograd            |     (371.492, 125)         |      (492.526, 215)        |      (80.542, 75)       |
| Tsitouras45, RungeKutta4, autograd         |     (388.067, 305)         |      (1023.593, 25)        |      (111.867, 235)     |
| euler, DormandPrince45, autograd           |     (397.314, 305)         |      (1005.025, 60)        |      (118.294, 385)     |
| DormandPrince45, Tsitouras45, adjoint      |     (409.832, 145)         |      (1003.121, 10)        |      (63.735, 25)       |
| DormandPrince45, midpoint, adjoint         |     (410.074, 125)         |      (1011.111, 155)       |      (63.968, 20)       |
| euler, midpoint, autograd                  |     (414.770, 145)         |      (1001.083, 25)        |      (118.695, 155)     |
| DormandPrince45, RungeKutta4, autograd     |     (418.147, 295)         |      (646.319, 160)        |      (82.971, 400)      |
| DormandPrince45, euler, autograd           |     (421.045, 175)         |      (637.637, 125)        |      (82.895, 400)      |
| Tsitouras45, RungeKutta4, adjoint          |     (426.851, 400)         |      (1040.776, 10)        |      (91.584, 380)      |
| Tsitouras45, DormandPrince45, autograd     |     (439.833, 120)         |      (1020.183, 15)        |      (100.023, 280)     |
| euler, RungeKutta4, autograd               |     (521.825, 370)         |      (1006.159, 25)        |      (100.135, 30)      |
| euler, Tsitouras45, autograd               |     (541.267, 380)         |      (1060.588, 20)        |      (100.123, 10)      |
| Tsitouras45, DormandPrince45, adjoint      |     (607.639, 250)         |      (1014.677, 20)        |      (122.620, 335)     |

<h1>Testing Loss Heuristics</h1>

| Experiment Type             |           one_run            |                all               |          next_only         |
|-----------------------------|:----------------------------:|:--------------------------------:|:--------------------------:|
| one_run_with_discount, adjoint|        (204.057, 10)       |             (479.940, 10)       |        (40.834, 15)       |
| all_with_discount, adjoint  |          (300.513, 90)      |            (491.818, 125)       |       (60.685, 235)       |
| all, adjoint                |          (306.988, 65)      |            (551.562, 290)       |       (75.726, 210)       |
| all_with_discount, autograd |          (320.790, 40)      |            (725.789, 100)       |        (43.146, 25)       |
| one_run, autograd           |         (340.196, 110)      |            (572.202, 115)       |       (71.155, 395)       |
| one_run_with_discount, autograd|     (344.422, 55)         |           (1095.972, 15)        |        (78.034, 30)       |
| one_run, adjoint            |         (415.506, 395)      |           (1004.670, 55)        |       (81.891, 400)       |
| all, autograd               |          (502.975, 90)      |            (559.443, 90)        |      (108.473, 390)       |
| next_only, adjoint          |           (1035.950, 25)    |             (1422.917, 20)      |        (67.715, 30)       |
| next_only, autograd         | (1e17, 215)| (1e18, 270) |       (64.409, 120)       |

