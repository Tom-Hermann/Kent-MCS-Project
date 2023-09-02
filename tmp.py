import svgwrite
import math

def layer_def(type, height, id=None):
    return {
        'type': type,
        'height': height,
        'id': id
    }

class CNNDraw:
    def __init__(self, dwg):
        self.__dwg = dwg

    def draw_arrow(self, p_points, width=2, color='black'):
        marker = self.__dwg.marker(insert=(2.1, 2), size=(2, 4), orient='auto')
        marker.add(self.__dwg.path(d='M0,0 V4 L2,2 Z', fill=color))
        self.__dwg.defs.add(marker)
        line = self.__dwg.add(svgwrite.shapes.Polyline(
            p_points, stroke_width=width,
            stroke='black', fill='none'))
        line.set_markers((self.__dwg.marker(), self.__dwg.marker(), marker))

    def draw_3d_rectangle(self, p_start, width, height, stretch, fg_color='white', bg_color='grey', border_color='black'):
        x_start, y_start = p_start
        x_end, y_end = x_start + width, y_start + height

        polygon1 = [(x_start, y_start), (x_start + width, y_start), (x_start + width, y_start + height), (x_start, y_start + height)]
        polygon2 = [(x_start, y_start), (x_start + stretch, y_start - stretch), (x_end + stretch, y_start - stretch), (x_end, y_start)]
        polygon3 = [(x_end, y_start), (x_end + stretch, y_start - stretch), (x_end + stretch, y_end - stretch), (x_end, y_end)]

        self.__dwg.add(svgwrite.shapes.Polygon(polygon1, fill=fg_color, stroke=border_color))
        self.__dwg.add(svgwrite.shapes.Polygon(polygon2, fill=bg_color, stroke=border_color))
        self.__dwg.add(svgwrite.shapes.Polygon(polygon3, fill=bg_color, stroke=border_color))

    def draw_multiple_layers(self, p_start, p_stretch=0.4, layer_width=10, space_width=7, layers=[], return_hash=True): # layers = (height, color)
        if len(layers) == 0:
            return
        max_height = max(map(lambda l: l[1], filter(lambda l: isinstance(l, tuple), layers)))
        x_offset = 0
        side_width_before = 0
        drawn_polygons = []
        for n in range(len(layers)):
            layer = layers[n]
            if not isinstance(layer, tuple):
                x_offset += layer
                continue

            id, height, fg_color, bg_color = layer
            side_width = math.floor(height * p_stretch)
            total_width = side_width + layer_width

            if n > 0:
                space_offset = space_width
                if total_width < side_width_before:
                    space_offset = math.floor((side_width_before - total_width) * 0.5)
                    space_offset = max(space_width, space_offset)
                    pass
                x_offset += space_offset

            curr_p_start = (p_start[0] + x_offset, p_start[1] + math.floor((max_height - height) * 0.5))
            self.draw_3d_rectangle(curr_p_start, layer_width, height, side_width, fg_color, bg_color)
            drawn_polygons.append((id, curr_p_start, layer_width, height, side_width))

            side_width_before = side_width
            x_offset += layer_width
        if return_hash:
            polygons = {}
            for polygon in filter(lambda l: l[0] is not None, drawn_polygons):
                polygons[polygon[0]] = polygon[1:]
            drawn_polygons = polygons
        return drawn_polygons

    def draw_multiple_defined_layers(self, p_start, layer_definitions={}, layers=[]):
        plain_layers = []
        for layer in layers:
            if not isinstance(layer, dict):
                plain_layers.append(layer)
            else:
                layer_definition = layer_definitions[layer['type']]
                if 'ignore' in layer_definition and layer_definition['ignore']:
                    continue
                plain_layer = (layer['id'], layer['height'], layer_definition['fg_color'], layer_definition['bg_color'])
                if 'add_space_before' in layer_definition:
                    plain_layers.append(layer_definition['add_space_before'])
                plain_layers.append(plain_layer)
                if 'add_space_after' in layer_definition:
                    plain_layers.append(layer_definition['add_space_after'])
        return self.draw_multiple_layers(p_start, layers=plain_layers)

    def draw_arrow_between_points(self, source_point, target_point, text=None, y_delta=10, bottom=False):
        if not bottom:
            y = min(source_point[1], target_point[1]) - y_delta
        else:
            y = max(source_point[1], target_point[1]) + y_delta
        source_delta = source_point[1] - y
        target_delta = target_point[1] - y
        if not bottom:
            mp0 = (source_point[0] + source_delta, source_point[1] - source_delta)
            mp1 = (target_point[0] - target_delta, target_point[1] - target_delta)
        else:
            mp0 = (source_point[0] - source_delta, source_point[1] - source_delta)
            mp1 = (target_point[0] + target_delta, target_point[1] - target_delta)
        points = [
            source_point,
            mp0, mp1,
            target_point,
        ]
        self.draw_arrow(points)

        # If required: Draw the text
        if text is not None:
            text_lines = text.split('\n')
            line_height = 15
            mp_middle = (math.floor((mp0[0] + mp1[0]) * 0.5), math.floor((mp0[1] + mp1[1]) * 0.5))
            if not bottom:
                y_offset = -10 - line_height * (len(text_lines) - 1)
            else:
                y_offset = 13
            for text_line in text_lines:
                mp_text = (mp_middle[0], mp_middle[1] + y_offset)
                self.__dwg.add(self.__dwg.text(text_line, insert=mp_text, text_anchor="middle", style="font-family:Sans-Serif"))
                y_offset += line_height

    def draw_arrow_between_layers(self, source_layer, target_layer, text=None, y_delta=10):
        source_point = (source_layer[0][0] + source_layer[1] + source_layer[3], source_layer[0][1] - source_layer[3])
        target_point = (target_layer[0][0] + target_layer[3], target_layer[0][1] - target_layer[3])
        self.draw_arrow_between_points(source_point, target_point, text, y_delta)

    def draw_additive_arrow_between_layers(self, source_layers, target_layer, text=None, y_delta=10, y_add=0, bottom=True):
        if bottom:
            target_point = (target_layer[0][0], target_layer[0][1] + target_layer[2])

            # Get the source points
            sp0 = min(map(lambda l: (l[0][0] + math.floor(l[1] * 0.5), l[0][1] + l[2]), source_layers), key=lambda l: l[0])
            sp1 = max(map(lambda l: (l[0][0] + math.floor(l[1] * 0.5), l[0][1] + l[2]), source_layers), key=lambda l: l[0])
            sp_y = max(map(lambda l: l[0][1] + l[2], source_layers)) + 5

            sp0d = (sp0[0], sp_y + y_add)
            sp1d = (sp1[0], sp_y + y_add)
        else:
            target_point = (target_layer[0][0] + target_layer[3], target_layer[0][1] - target_layer[3])

            # Get the source points
            sp0 = min(map(lambda l: (l[0][0] + l[3] + math.floor(l[1] * 0.5), l[0][1] - l[3]), source_layers), key=lambda l: l[0])
            sp1 = max(map(lambda l: (l[0][0] + l[3] + math.floor(l[1] * 0.5), l[0][1] - l[3]), source_layers), key=lambda l: l[0])
            sp_y = min(map(lambda l: l[0][1] - l[3], source_layers)) - 5

            sp0d = (sp0[0], sp_y - y_add)
            sp1d = (sp1[0], sp_y - y_add)

        self.__dwg.add(svgwrite.shapes.Polyline([sp0, sp0d], stroke_width=2, stroke='black', fill='none'))
        self.__dwg.add(svgwrite.shapes.Polyline([sp1, sp1d], stroke_width=2, stroke='black', fill='none'))

        if bottom:
            mp0 = (sp0d[0] + y_delta, sp0d[1] + y_delta)
            mp1 = (sp1d[0] - y_delta, sp1d[1] + y_delta)
        else:
            mp0 = (sp0d[0] + y_delta, sp0d[1] - y_delta)
            mp1 = (sp1d[0] - y_delta, sp1d[1] - y_delta)

        mp_middle = (math.floor((mp0[0] + mp1[0]) * 0.5), math.floor((mp0[1] + mp1[1]) * 0.5))

        # Draw the first line
        self.__dwg.add(svgwrite.shapes.Polyline(
            [sp0d, mp0, mp1, sp1d], stroke_width=2,
            stroke='black', fill='none'))

        # And now the line itself
        self.draw_arrow_between_points(mp_middle, target_point, text, y_delta, bottom=bottom)

if __name__ == '__main__':
    dwg = svgwrite.Drawing('test.svg')
    cnn_draw = CNNDraw(dwg)

    # Draw multiple layers
    layer_polygons = cnn_draw.draw_multiple_defined_layers((100, 200), { # Start coordinate

        # Define several layer types
        'dropout': {'fg_color': '#ffffcc', 'bg_color': '#ffff99', 'ignore': True},
        'conv': {'fg_color': '#e6faff', 'bg_color': '#99ebff'},
        't_conv': {'fg_color': '#ffe6cc', 'bg_color': '#ffcc99'},
        'pool': {'fg_color': '#ccffcc', 'bg_color': '#99ff99', 'add_space_after': 20},
        'upscale': {'fg_color': '#e6ccff', 'bg_color': '#cc99ff'}

    }, [

        # The layers to draw; as can be seen a layer can have an id
        layer_def('dropout', 300),
        layer_def('conv', 300),
        layer_def('conv', 300),
        layer_def('pool', 300, id='POOL1'),

        layer_def('dropout', 250),
        layer_def('conv', 250),
        layer_def('conv', 250),
        layer_def('pool', 250, id='POOL2'),

        layer_def('dropout', 200),
        layer_def('conv', 200),
        layer_def('conv', 200),
        layer_def('pool', 200, id='POOL3'),

        layer_def('dropout', 150),
        layer_def('conv', 150),
        layer_def('conv', 150),
        layer_def('pool', 150, id='POOL4'),

        layer_def('dropout', 100),
        layer_def('conv', 100),
        layer_def('conv', 100),
        layer_def('pool', 100, id='POOL5'),

        layer_def('dropout', 50),
        layer_def('conv', 50),
        layer_def('conv', 50),
        layer_def('pool', 50, id='POOL6'),

        layer_def('dropout', 30),
        layer_def('conv', 30, id='FINAL'),

        # Plain numbers are just space
        100,
        layer_def('t_conv', 50, id='T_CONV1'),
        60,
        layer_def('t_conv', 100, id='T_CONV2'),
        60,
        layer_def('t_conv', 150, id='T_CONV3'),
        60,
        layer_def('t_conv', 250, id='T_CONV4'),

        layer_def('conv', 250),
        layer_def('conv', 250),
        layer_def('conv', 250),
        layer_def('dropout', 250),
        layer_def('conv', 250),
        layer_def('conv', 250, id='FINAL_CONV'),

        50,
        layer_def('upscale', 300, id='UPSCALED')
    ])

    cnn_draw.draw_arrow_between_layers(layer_polygons['FINAL'], layer_polygons['T_CONV1'], text="transposed conv.")
    cnn_draw.draw_additive_arrow_between_layers([layer_polygons['POOL5'], layer_polygons['T_CONV1']], layer_polygons['T_CONV2'], text="element-wise sum and\ntransposed conv.", y_add=0)
    cnn_draw.draw_additive_arrow_between_layers([layer_polygons['POOL4'], layer_polygons['T_CONV2']], layer_polygons['T_CONV3'], text="element-wise sum and\ntransposed conv.", y_add=0, bottom=False)
    cnn_draw.draw_additive_arrow_between_layers([layer_polygons['POOL3'], layer_polygons['T_CONV3']], layer_polygons['T_CONV4'], text="element-wise sum and\ntransposed conv.", y_add=0)
    cnn_draw.draw_arrow_between_layers(layer_polygons['FINAL_CONV'], layer_polygons['UPSCALED'], text="upscale")

    dwg.save()