import cv2


def cv2_plot_lines(frame, pts, order):
    color_mapping = {1: [255, 0, 255], 2: [255, 0, 0], 3: [255, 0, 127], 4: [255, 255, 255], 5: [0, 0, 255],
                     6: [0, 127, 255], 7: [0, 255, 255], 8: [0, 255, 0], 9: [200, 162, 200]}
    # point_size = 7
    point_size = 2
    if order == 0:
        # other animals
        # plot neck-eyes
        cv2.line(frame, (pts[[0, 2], :][0, 0], pts[[0, 2], :][0, 1]), (pts[[0, 2], :][1, 0], pts[[0, 2], :][1, 1]),
                 color_mapping[1], point_size)
        cv2.line(frame, (pts[[1, 2], :][0, 0], pts[[1, 2], :][0, 1]), (pts[[1, 2], :][1, 0], pts[[1, 2], :][1, 1]),
                 color_mapping[2], point_size)

        # plot legs
        cv2.line(frame, (pts[[3, 8], :][0, 0], pts[[3, 8], :][0, 1]), (pts[[3, 8], :][1, 0], pts[[3, 8], :][1, 1]),
                 color_mapping[5], point_size)
        cv2.line(frame, (pts[[8, 14], :][0, 0], pts[[8, 14], :][0, 1]), (pts[[8, 14], :][1, 0], pts[[8, 14], :][1, 1]),
                 color_mapping[5], point_size)

        cv2.line(frame, (pts[[4, 9], :][0, 0], pts[[4, 9], :][0, 1]), (pts[[4, 9], :][1, 0], pts[[4, 9], :][1, 1]),
                 color_mapping[6], point_size)
        cv2.line(frame, (pts[[9, 15], :][0, 0], pts[[9, 15], :][0, 1]), (pts[[9, 15], :][1, 0], pts[[9, 15], :][1, 1]),
                 color_mapping[6], point_size)

        cv2.line(frame, (pts[[5, 10], :][0, 0], pts[[5, 10], :][0, 1]), (pts[[5, 10], :][1, 0], pts[[5, 10], :][1, 1]),
                 color_mapping[7], point_size)
        cv2.line(frame, (pts[[10, 16], :][0, 0], pts[[10, 16], :][0, 1]),
                 (pts[[10, 16], :][1, 0], pts[[10, 16], :][1, 1]), color_mapping[7], point_size)

        cv2.line(frame, (pts[[6, 11], :][0, 0], pts[[6, 11], :][0, 1]), (pts[[6, 11], :][1, 0], pts[[6, 11], :][1, 1]),
                 color_mapping[8], point_size)
        cv2.line(frame, (pts[[11, 17], :][0, 0], pts[[11, 17], :][0, 1]),
                 (pts[[11, 17], :][1, 0], pts[[11, 17], :][1, 1]), color_mapping[8], point_size)

        # plot hip-necks
        cv2.line(frame, (pts[[12, 7], :][0, 0], pts[[12, 7], :][0, 1]), (pts[[12, 7], :][1, 0], pts[[12, 7], :][1, 1]),
                 color_mapping[1], point_size)
        cv2.line(frame, (pts[[13, 7], :][0, 0], pts[[13, 7], :][0, 1]), (pts[[13, 7], :][1, 0], pts[[13, 7], :][1, 1]),
                 color_mapping[2], point_size)
    elif order == 1:
        # elephant
        cv2.line(frame, (pts[[0, 2], :][0, 0], pts[[0, 2], :][0, 1]), (pts[[0, 2], :][1, 0], pts[[0, 2], :][1, 1]),
                 color_mapping[1], point_size)
        cv2.line(frame, (pts[[1, 2], :][0, 0], pts[[1, 2], :][0, 1]), (pts[[1, 2], :][1, 0], pts[[1, 2], :][1, 1]),
                 color_mapping[2], point_size)

        # plot legs
        cv2.line(frame, (pts[[3, 8], :][0, 0], pts[[3, 8], :][0, 1]), (pts[[3, 8], :][1, 0], pts[[3, 8], :][1, 1]),
                 color_mapping[5], point_size)
        cv2.line(frame, (pts[[8, 14], :][0, 0], pts[[8, 14], :][0, 1]), (pts[[8, 14], :][1, 0], pts[[8, 14], :][1, 1]),
                 color_mapping[5], point_size)

        cv2.line(frame, (pts[[4, 9], :][0, 0], pts[[4, 9], :][0, 1]), (pts[[4, 9], :][1, 0], pts[[4, 9], :][1, 1]),
                 color_mapping[6], point_size)
        cv2.line(frame, (pts[[9, 15], :][0, 0], pts[[9, 15], :][0, 1]), (pts[[9, 15], :][1, 0], pts[[9, 15], :][1, 1]),
                 color_mapping[6], point_size)

        cv2.line(frame, (pts[[5, 10], :][0, 0], pts[[5, 10], :][0, 1]), (pts[[5, 10], :][1, 0], pts[[5, 10], :][1, 1]),
                 color_mapping[7], point_size)
        cv2.line(frame, (pts[[10, 16], :][0, 0], pts[[10, 16], :][0, 1]),
                 (pts[[10, 16], :][1, 0], pts[[10, 16], :][1, 1]), color_mapping[7], point_size)

        cv2.line(frame, (pts[[6, 11], :][0, 0], pts[[6, 11], :][0, 1]), (pts[[6, 11], :][1, 0], pts[[6, 11], :][1, 1]),
                 color_mapping[8], point_size)
        cv2.line(frame, (pts[[11, 17], :][0, 0], pts[[11, 17], :][0, 1]),
                 (pts[[11, 17], :][1, 0], pts[[11, 17], :][1, 1]), color_mapping[8], point_size)

        # plot hip-necks
        cv2.line(frame, (pts[[12, 7], :][0, 0], pts[[12, 7], :][0, 1]), (pts[[12, 7], :][1, 0], pts[[12, 7], :][1, 1]),
                 color_mapping[1], point_size)
        cv2.line(frame, (pts[[13, 7], :][0, 0], pts[[13, 7], :][0, 1]), (pts[[13, 7], :][1, 0], pts[[13, 7], :][1, 1]),
                 color_mapping[2], point_size)

        cv2.line(frame, (pts[[18, 19], :][0, 0], pts[[18, 19], :][0, 1]),
                 (pts[[18, 19], :][1, 0], pts[[18, 19], :][1, 1]), color_mapping[1], point_size)
        cv2.line(frame, (pts[[19, 20], :][0, 0], pts[[19, 20], :][0, 1]),
                 (pts[[19, 20], :][1, 0], pts[[19, 20], :][1, 1]), color_mapping[1], point_size)
        cv2.line(frame, (pts[[20, 21], :][0, 0], pts[[20, 21], :][0, 1]),
                 (pts[[20, 21], :][1, 0], pts[[20, 21], :][1, 1]), color_mapping[1], point_size)
        cv2.line(frame, (pts[[21, 22], :][0, 0], pts[[21, 22], :][0, 1]),
                 (pts[[21, 22], :][1, 0], pts[[21, 22], :][1, 1]), color_mapping[1], point_size)
        cv2.line(frame, (pts[[22, 23], :][0, 0], pts[[22, 23], :][0, 1]),
                 (pts[[22, 23], :][1, 0], pts[[22, 23], :][1, 1]), color_mapping[1], point_size)
        cv2.line(frame, (pts[[23, 24], :][0, 0], pts[[23, 24], :][0, 1]),
                 (pts[[23, 24], :][1, 0], pts[[23, 24], :][1, 1]), color_mapping[1], point_size)


def cv2_visualize_keypoints(frame, pts, num_pts=18, order=0):
    points = pts
    x = []
    y = []
    for i in range(num_pts):
        x.append(points[i][0])
        y.append(points[i][1])
        # plot keypoints on each image
        cv2.circle(frame, (x[-1], y[-1]), 2, (0, 255, 0), -1)
    cv2_plot_lines(frame, points, order)
    return frame