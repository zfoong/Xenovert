"""
@author: Danilo Vasconcellos Vargas, Tham Yik Foong 
"""
import numpy as np

class xenovert:
    
    def __init__(self, learning_rate, level=0, round_to=None, init_value=0):
        self.learning_rate = learning_rate
        self.value= init_value
        self.intrinsic_value = 0
        self.velocity = 0
        self.right= None
        self.left= None
        self.level= level
        self.round_to = round_to
    
    def convert(self, value, offset=0):
        """
        Convert Value into Interval  (Quantized output)

        Parameters
        ----------
        value : 
            Input x at time t.
        offset : optional
            Offset for level in tree. The default is 0.

        Returns
        -------
        TYPE
            Quantized output.

        """
        if self.left is not None:
            if value < self.value:
                return self.left.convert(value, offset)
            else:
                return self.right.convert(value, 2**(self.level)+offset)
        else:
            output= np.zeros(2**(self.level+1))
            if value < self.value:
                output[offset]= 1
            else:
                output[offset+2**self.level]= 1

            return output

    def input(self, value):
        """
        Input for adaptation

        Parameters
        ----------
        value : 
            Input x at time t.

        Returns
        -------
        None.

        """
        self.velocity *= 0.99
        self.velocity += np.abs(self.value - value)
        
        switch = 1 
        if (self.value - value) > 0:
            switch = -1

        self.value += self.learning_rate*self.velocity*switch
        
        if self.left is not None:
            if value < self.value:
                self.left.input(value)
            else:
                self.right.input(value)

    def grow(self):
        """
        Grow the binary tree by adding one level to the tree

        Returns
        -------
        None.

        """
        if self.left is not None:
            self.left.grow()
            self.right.grow()
        else:
            lr = self.learning_rate
            self.left= xenovert(lr,self.level+1, round_to=self.round_to, init_value=self.value)
            self.right= xenovert(lr,self.level+1, round_to=self.round_to, init_value=self.value)
            
    def return_val(self):
        value_list = []
        if self.left is not None:
            val1 = self.right.return_val()
            val2 = self.left.return_val()
            for i in val1:
                value_list.append(i)
            for i in val2:
                value_list.append(i)
            value_list.append(self.value)
            return value_list
        else:
            value_list.append(self.value)
            return value_list
        
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