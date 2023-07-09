
import numpy as np
import matplotlib.pyplot as plt
import math

class xenovert_2D:
    
    def __init__(self, learning_rate, level=0, round_to=None, init_value=[0,0], slope=1, y_intercept=0, s_learning_rate=0.00001, parent=None, direction='', poly_points=[[-999,-999], [999,-999], [999,999], [-999,999]]):
        self.aging_coefficient= 0.1
        self.learning_rate = learning_rate
        self.value = init_value
        self.intrinsic_value = 0
        self.velocity = np.array([0.0,0.0])
        self.right= None
        self.left= None
        self.level= level
        self.round_to = round_to
        self.slope = slope
        self.y_intercept = y_intercept
        self.s_learning_rate = s_learning_rate
        self.parent = parent
        self.direction = direction
        self.poly_points = poly_points
        self.slope_toggle = False
        self.prev_slope = slope

    def aging(self):
        self.learning_rate*= self.aging_coefficient
        if self.left is not None:
            self.left.aging()
            self.right.aging()
    
    def frozen(self):
        self.learning_rate= 0
        if self.left is not None:
            self.left.aging()
            self.right.aging()
    
    def convert(self, value, offset=0):
        
        line_x = self.line_x(value[1], self.slope, self.y_intercept)
        line_y = self.line(value[0], self.slope, self.y_intercept)
        
        if self.left.left is not None:
            
            if self.slope_toggle:
                if value[0] >= line_x:
                    return self.left.convert(value, offset)
                else:
                    return self.right.convert(value, 2**(self.level)+offset)
            else:
                if value[0] < line_x:
                    return self.left.convert(value, offset)
                else:
                    return self.right.convert(value, 2**(self.level)+offset)
            
        else:
            output= np.zeros(2**(self.level+1))

            if self.slope_toggle:
                if value[0] >= line_x:
                    output[offset]= 1
                else:
                    output[offset+2**self.level]= 1
            else:
                if value[0] < line_x:
                    output[offset]= 1
                else:
                    output[offset+2**self.level]= 1
            return output
        
    def line(self, x, m, c):
        return m * x + c
    
    def line_x(self, y, m, c):
        return (y - c) / m
    
    def update_slope(self):
        target = self.slope
        if self.level % 2 == 0:
            x1, y1 = self.left.value[0], self.left.value[1]
            x2, y2 = self.right.value[0], self.right.value[1]
            
            # Assign a small value to detect when to 'flip' the line
            if (x2-x1) <= 0.000001 and (x2-x1) > -0.000001:
                return self.slope
            
            m=(y1-y2)/(x1-x2)
            target = -1/m
        else:
            target = -1/self.parent.slope
 
        if self.slope * target == -1:
            tan=(self.slope-target)
        else:
            tan=(self.slope-target)/(1+target*self.slope)
            
        self.slope=np.tan(np.arctan(self.slope)-0.001*np.arctan(tan))
    
    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        
        def within_bound(a, b):
            if (a >= line2[0][0] and a <= line2[1][0]) or (a >= line2[1][0] and a <= line2[0][0]):
                if (b >= line2[0][1] and b <= line2[1][1]) or (b >= line2[1][1] and b <= line2[0][1]):
                    return True
                if (line2[1][1] == line2[0][1]):
                    return True
            if line2[0][0] == line2[1][0]:
                if (b >= line2[0][1] and b <= line2[1][1]) or (b >= line2[1][1] and b <= line2[0][1]):
                    return True
                if (line2[1][1] == line2[0][1]):
                    return True
            return False
    
        div = det(xdiff, ydiff)
        if div == 0:
           print('lines do not intersect')
           return None
    
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        
        if within_bound(x,y) is False:
            return None
        
        return [x, y]

    def input(self, value):
        self.velocity *= 0.99
        self.velocity += np.abs(self.value - value)
        self.value += self.learning_rate*(value - self.value) / np.abs(self.value - value)*self.velocity
        
        self.y_intercept = self.value[1] - self.slope * self.value[0]
        line_x = self.line_x(value[1], self.slope, self.y_intercept)
        line_y = self.line(value[0], self.slope, self.y_intercept)
                
        if self.left is not None:
            self.prev_slope = self.slope
            self.update_slope()
            if self.slope <= 0.1 and self.slope >= -0.1:
                if (self.prev_slope >= 0 and self.slope < 0) or (self.prev_slope <= 0 and self.slope > 0):
                    if self.slope_toggle is True:
                        self.slope_toggle = False
                    else:               
                        self.slope_toggle = True
                
            if self.slope_toggle:
                if value[0] >= line_x:
                    self.left.input(value)
                else:
                    self.right.input(value)
            else:
                if value[0] < line_x:
                    self.left.input(value)
                else:
                    self.right.input(value)
            
    def update_all_slope(self):
        if self.left is not None:
            self.left.update_all_slope()
            self.right.update_all_slope()
                
    def area_subdivision(self):            
        lines = []
        
        if self.level == 0:
            self.poly_points = [[self.value[0]-1999,self.value[1]-999], 
                                [self.value[0]+1999,self.value[1]-999], 
                                [self.value[0]+1999,self.value[1]+999], 
                                [self.value[0]-1999,self.value[1]+999]]
        
        for i in range(len(self.poly_points)):
            if i == (len(self.poly_points)-1):
                lines.append([self.poly_points[i],self.poly_points[0]])
            else:
                lines.append([self.poly_points[i],self.poly_points[i+1]])
        
        self_line_min = [self.line_x(self.value[1]-100000, self.slope, self.y_intercept), self.value[1]-100000]
        self_line_max = [self.line_x(self.value[1]+100000, self.slope, self.y_intercept), self.value[1]+100000]
        self_line = [self_line_min, self_line_max]
        
        main_lines_intersect = []
        intersect_id = []
        
        for line_id, line in enumerate(lines):
            intersect = self.line_intersection(self_line, line)
            main_lines_intersect.append(intersect)
            if intersect is not None:
                intersect_id.append(line_id)
            
        main_lines_intersect = [i for i in main_lines_intersect if i]
        
        if len(intersect_id) > 1:
            
            included_points_id = []
            excluded_points_id = []
            included_points = []
            excluded_points = []

            max_intersect_id = max(intersect_id)
            id_range = intersect_id[1] - intersect_id[0]
            
            included_points_id = [i for i in range(max_intersect_id-id_range+1, max_intersect_id+1)]
            
            excluded_points_id = [i for i in range(len(self.poly_points))]
            for i in included_points_id:
                excluded_points_id.remove(i)
            
            for i in included_points_id:
                included_points.append(self.poly_points[i])
                
            for i in excluded_points_id:
                excluded_points.append(self.poly_points[i])
                
            for p in main_lines_intersect:
                if p is not None:
                    included_points.append(p)
                    excluded_points.append(p)
            
            cent=(sum([p[0] for p in included_points])/len(included_points),sum([p[1] for p in included_points])/len(included_points))
            included_points.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
            
            cent=(sum([p[0] for p in excluded_points])/len(excluded_points),sum([p[1] for p in excluded_points])/len(excluded_points))
            excluded_points.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
            

            if self.left is not None:    
                self.left.set_poly_points(included_points)
                self.right.set_poly_points(excluded_points)
                    
        if self.left is not None:
            self.left.area_subdivision()
            self.right.area_subdivision()
                
    def set_poly_points(self, x):
        self.poly_points = x
        return
    
    def return_area_points(self):
        poly_points_list = []
        if self.left is None:
            poly_points_list.append(self.poly_points)
            return poly_points_list
        else:
            fpr = self.right.return_area_points()
            fpl = self.left.return_area_points()
            for i in fpr:
                poly_points_list.append(i)
            for i in fpl:
                poly_points_list.append(i)
            return poly_points_list

    def grow(self):
        if self.left is not None:
            self.left.grow()
            self.right.grow()
        else:
            lr = self.learning_rate*2
            slr = self.s_learning_rate*2
            l_init_value = [self.value[0]-(1/10*self.level), self.value[1]-(1/10*self.level)]
            r_init_value = [self.value[0]+(1/10*self.level), self.value[1]+(1/10*self.level)]
            self.left= xenovert_2D(lr,self.level+1, round_to=self.round_to, init_value=l_init_value, slope=-1/self.slope, parent=self, direction='l', s_learning_rate = slr)
            self.right= xenovert_2D(lr,self.level+1, round_to=self.round_to, init_value=r_init_value, slope=-1/self.slope, parent=self, direction='r', s_learning_rate = slr)
            
    def return_val(self):
        value_list = []
        slope_list = []
        if self.left.left is not None:
            val1, s1 = self.right.return_val()
            val2, s2 = self.left.return_val()
            for i in val1:
                value_list.append(i)
            for i in val2:
                value_list.append(i)
            for i in s1:
                slope_list.append(i)
            for i in s2:
                slope_list.append(i)
            value_list.append(self.value)
            slope_list.append(self.slope)
            return value_list, slope_list
        else:
            value_list.append(self.value)
            slope_list.append(self.slope)
            return value_list, slope_list
        
        
    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.value
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.value
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2