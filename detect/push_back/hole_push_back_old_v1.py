import cv2
import numpy as np
import math
import tool.stand_tool
import random

class HolePushBackModel:

    def __init__(self, hole_push_bach_config : dict, stand_tool : tool.stand_tool.StandTool ) -> None:

        self._hole_push_back_config = hole_push_bach_config
        self._stand_tool = None

        self._stand_shape = None
        self._stand_hole_detail = None
        self._stand_length_relationship = None

        self._push_back_vector = None


        self.set_stand_tool( stand_tool )

    def set_stand_tool(self, stand_tool: tool.stand_tool.StandTool) -> None:

        self._stand_tool = stand_tool

        self._stand_shape = self._stand_tool.get_shape()
        self._stand_hole_detail = self._stand_tool.get_hole_detail()
        # self._stand_length_relationship : list  = self._stand_tool.get_length_relationship()
        self._stand_length_relationship : list  = self._stand_tool.get_length_relationship_convert()

    
    def _cal_angle(self, v1, v2):

        norm = np.linalg.norm(v1) * np.linalg.norm(v2)

        rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / norm))

        theta = np.rad2deg(np.arccos(np.dot(v1, v2) / norm))

        if math.isnan(theta):

            theta = 0


        if rho < 0:
            
            return - theta
        
        else:
        
            return theta
    
    
    def _cal_ratio(self, v1, v2):

        length1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        length2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

        return length2 / length1


    def _is_angle_match(self, angle, match_angle):

        difference_value = self._hole_push_back_config['angle_match_deviation']

        if angle < 0 and match_angle < 0 :

            if match_angle * ( 1 - difference_value ) >= angle and angle >= match_angle * ( 1 + difference_value ) :

                return True

        elif match_angle * ( 1 - difference_value ) <= angle and angle <= match_angle * ( 1 + difference_value ) :

            return True

        return False

    def _is_ratio_match(self, ratio, match_ratio):

        difference_value = self._hole_push_back_config['ratio_match_deviation']

        if match_ratio * ( 1 - difference_value ) <= ratio and ratio <= match_ratio * ( 1 + difference_value ) :

            return True

        return False


    def _length_create(self, hole_detail : list) -> (list, list):

        start_hole = []
        length_detail = []

        for hole1 in hole_detail:

            _, middle1 = hole1

            pos1 = hole_detail.index(hole1) + 1

            length_vector_list = []

            for hole2 in hole_detail:

                if hole1 == hole2: continue

                _, middle2 = hole2

                pos2 = hole_detail.index(hole2) + 1

                length_vector = []

                v = (middle2[0] - middle1[0], middle2[1] - middle1[1])

                length_vector.append(str(pos1))
                length_vector.append(str(pos2))
                length_vector.append(v)
                length_vector.append((v[0] ** 2 + v[1] ** 2) ** 0.5)

                length_vector_list.append(length_vector)

            start_hole.append(middle1)
            length_detail.append(length_vector_list)

        return start_hole, length_detail


    def _get_first_length_possible_combination(self, length_detail) -> list:
        
        # length 1
        l1 = length_detail[0]

        start1, end1, length_vector1, length1 = l1

        match_list = []
        start_length_match_list = []

        # length 2
        for l2 in length_detail[1:]:

            start2, end2, length_vector2, length2 = l2

            angle = self._cal_angle(length_vector1, length_vector2)

            ratio = length2 / length1

            match_list = [x for x in self._stand_length_relationship
                          if self._is_angle_match(angle, x[3]) and
                          self._is_ratio_match(ratio, x[4])]


            # find match combination for length 1 and 2
            min_match = []
            min_match_angle = 100
            min_match_ratio = 100

            for match in match_list:

                if abs(match[3] - angle) < min_match_angle and abs(match[4] - min_match_ratio) < min_match_ratio:

                    min_match = match
                    min_match_angle = abs(match[3] - angle)
                    min_match_ratio = abs(match[4] - min_match_ratio)

            if min_match:

                match_list.append(min_match)
                start_length_match_list.append([int(min_match[0]), int(min_match[1][0])])

        # possible combination for length 1
        count_dict = {}

        for item in start_length_match_list :

            if str(item) in count_dict:

                count_dict[str(item)] += 1

            else:

                count_dict[str(item)] = 1

        return list(count_dict.keys())

    def _get_push_back_info(self, first_length_info : list) -> list:

        start_length = first_length_info[1:-1]

        start_length = start_length.split(',')

        start_length = [start_length[0].strip(), start_length[1].strip()]

        push_back_info = [x for x in self._stand_length_relationship if
                          x[0] == start_length[0] and x[1][0] == start_length[1]]

        return push_back_info


    def _get_shape_proportion(self, real_shape : tuple) -> float:

        return ( max(real_shape) / max(self._stand_shape) +  min(real_shape) / min(self._stand_shape) ) / 2


    def _hole_match_count(self, push_back_list : list, hole_list : list) -> (int, int, list):

        allowance = self._hole_push_back_config['hole_match_deviation']

        hole_list = hole_list.copy()

        match_list = []
        min_hit = 2000

        for tag, push_back_pos in push_back_list:

            for hole_range, middle in hole_list:

                if push_back_pos[0] in range(int(hole_range[0][0] * (1 - allowance)) , int(hole_range[1][0] * (1 - allowance))+ 1) and \
                        push_back_pos[1] in range(int(hole_range[0][1] * (1 + allowance)) , int(hole_range[1][1] * (1 + allowance)) + 1):

                    match_list.append([tag, push_back_pos, hole_range])

                    if int(tag) < min_hit:

                        min_hit = int(tag)

                    hole_list.remove([hole_range, middle])

                    break

        return len(match_list), min_hit, match_list


    def get_push_back_position(self, image_shape : tuple, hole_detail : list, magnification_again : float = 1):

        if len(hole_detail) < 3:

            return [], 1

        # use hole info to create length info
        start_hole_list, length_detail_list = self._length_create(hole_detail)

        final_max_hit = 0
        final_max_hit_min_hole = 0
        final_max_hit_push_back_pos = []

        for i in range(0, self._hole_push_back_config['try_time']):

            # choose one length be first length
            # start_hole = random.choice(start_hole_list)
            start_hole = start_hole_list[0]
            length_detail = length_detail_list[start_hole_list.index(start_hole)]

            first_length = length_detail[0][3]
            first_length_vector = length_detail[0][2]

            max_hit = 0
            max_hit_min_hole = 0
            max_hit_push_back_pos = []

            # get possible combination for first length
            first_length_possible_combination = self._get_first_length_possible_combination(length_detail)

            # push back
            for first_length_info in first_length_possible_combination:

                push_back_info = self._get_push_back_info(first_length_info)

                stand_first_length = ( push_back_info[0][2][0][0] ** 2 + push_back_info[0][2][0][1] ** 2 ) ** 0.5

                shape_proportion = self._get_shape_proportion(image_shape)

                magnification = first_length / ( stand_first_length * shape_proportion ) * magnification_again

                rotate_angle = self._cal_angle(push_back_info[0][2][0], first_length_vector)

                convert_vector = []

                for detail in push_back_info:

                    tag = detail[1][1]
                    vector = detail[2][1]

                    v1 = (vector[0] * math.cos(math.radians(rotate_angle)) - vector[1] * math.sin(math.radians(rotate_angle)))
                    v2 = (vector[0] * math.sin(math.radians(rotate_angle)) + vector[1] * math.cos(math.radians(rotate_angle)))

                    v1 = int(v1 * magnification * shape_proportion)
                    v2 = int(v2 * magnification * shape_proportion)

                    convert_vector.append([tag, (v1 + start_hole[0], v2 + start_hole[1])])

                tag = push_back_info[0][1][0]
                vector = push_back_info[0][2][0]

                v1 = (vector[0] * math.cos(math.radians(rotate_angle)) - vector[1] * math.sin(math.radians(rotate_angle)))
                v2 = (vector[0] * math.sin(math.radians(rotate_angle)) + vector[1] * math.cos(math.radians(rotate_angle)))

                v1 = int(v1 * magnification * shape_proportion)
                v2 = int(v2 * magnification * shape_proportion)

                convert_vector.append([tag, (v1 + start_hole[0], v2 + start_hole[1])])

                tag = push_back_info[0][0]

                convert_vector.append([tag, (start_hole[0], start_hole[1])])

                hit_count, min_hole_hit, match_list = self._hole_match_count(convert_vector, hole_detail)

                if hit_count > max_hit:

                    max_hit = hit_count
                    max_hit_min_hole = min_hole_hit
                    max_hit_push_back_pos = convert_vector

                if hit_count > self._stand_tool.get_hole_count() * 0.65:

                    print("B1")

                    break

            if max_hit > final_max_hit :

                final_max_hit = max_hit
                final_max_hit_min_hole = max_hit_min_hole
                final_max_hit_push_back_pos = max_hit_push_back_pos

            if final_max_hit > self._stand_tool.get_hole_count() * 0.85:

                print("B2")

                break

        if final_max_hit < self._stand_tool.get_hole_count() / 2:

            final_max_hit_push_back_pos = []
            final_max_hit_min_hole = 0

            print("no match")

        print(final_max_hit)

        return final_max_hit_push_back_pos, final_max_hit_min_hole


    def get_hole_frame(self, frame : np.ndarray, push_back_hole : list, min_hit : int):

        frame = frame.copy()

        x_move = 25
        y_move = -25

        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

        for tag, pos in push_back_hole:

            if int(tag) >= min_hit:

                tag = tag.zfill(2)
                color = (50, 108, 66)

            else:

                tag = tag.zfill(2) + " - Done"
                color = (46, 137, 255)

            cv2.circle(frame, pos, 5, color, -1)


            p1, p2 = ((int(pos[0]) + x_move), (int(pos[1]) + y_move)), ((int(pos[0]) - x_move), (int(pos[1]) - y_move))

            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(str(tag), 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(frame, tag, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                        lw / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

        return frame

