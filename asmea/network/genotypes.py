from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_7x1_1x7',
    'conv_1x1_3x3',
]

plotcell=Genotype(normal=[('5',0),('8', 0),
    ('7', 1),('6', 1),
    ('0', 0),('6', 0),],normal_concat = [2,3, 4],
    reduce=[('4',0),('3', 0),
    ('6', 1),('3', 1),
    ('0', 0),('6', 0),],reduce_concat=[2,3,4])





NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
                            ('skip_connect', 0), ('sep_conv_3x3', 1),
                            ('skip_connect', 0), ('sep_conv_3x3', 1),
                            ('sep_conv_3x3', 0),('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
                    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
                            ('skip_connect', 2), ('max_pool_3x3', 0),
                            ('max_pool_3x3', 0), ('skip_connect', 2),
                            ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                            ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                            ('sep_conv_3x3', 1), ('skip_connect', 0),
                            ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
                    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
                            ('skip_connect', 2), ('max_pool_3x3', 1),
                            ('max_pool_3x3', 0), ('skip_connect', 2),
                            ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

SNAS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                        ('skip_connect', 0), ('dil_conv_3x3', 1),
                        ('skip_connect', 1), ('skip_connect', 0), 
                        ('skip_connect',0),  ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
                reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
                       ('max_pool_3x3', 1), ('skip_connect', 2),
                       ('skip_connect', 2), ('max_pool_3x3', 1),
                       ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

PDARTS = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1),
                          ('skip_connect', 0),('sep_conv_3x3', 1),
                          ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
                          ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
                  reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1),
                          ('sep_conv_3x3', 0), ('dil_conv_5x5', 2),
                          ('max_pool_3x3', 0), ('dil_conv_3x3', 1),
                          ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

RNAS = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
                               ('sep_conv_3x3', 1), ('sep_conv_5x5', 1),
                               ('skip_connect', 0), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

PC_DARTS = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0),
                            ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
                            ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
                            ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
                    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0),
                            ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
                            ('sep_conv_3x3', 0),('sep_conv_3x3', 3),
                            ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

CDARTS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                          ('sep_conv_3x3', 1), ('sep_conv_3x3', 2),
                          ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                          ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
                  reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0),
                          ('dil_conv_5x5', 2), ('skip_connect', 0),
                          ('sep_conv_3x3', 0), ('sep_conv_5x5', 1),
                          ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

CARS_I = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                          ('skip_connect', 0), ('sep_conv_5x5', 1),
                          ('skip_connect', 2), ('sep_conv_3x3', 3),
                          ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
                  reduce=[('dil_conv_3x3', 0), ('skip_connect', 1),
                          ('max_pool_3x3', 0), ('max_pool_3x3', 2),
                          ('skip_connect', 1), ('sep_conv_5x5', 3),
                          ('dil_conv_3x3', 1), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

CARS_H = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
                          ('sep_conv_3x3', 0), ('dil_conv_5x5', 2),
                          ('avg_pool_3x3', 0), ('skip_connect', 1),
                          ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
                  reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1),
                          ('sep_conv_3x3', 0), ('skip_connect', 1),
                          ('dil_conv_3x3', 2), ('max_pool_3x3', 0),
                          ('sep_conv_5x5', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

DARTS = DARTS_V2


EEEA_B = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 0),
                          ('max_pool_3x3', 1), ('skip_connect', 1),
                          ('skip_connect', 1), ('skip_connect', 2),
                          ('dil_conv_3x3', 3), ('sep_conv_3x3', 1),
                          ('max_pool_3x3', 4), ('sep_conv_3x3', 4)], normal_concat=[5, 6],
                  reduce=[('skip_connect', 0), ('inv_res_5x5', 0),
                          ('avg_pool_3x3', 0), ('dil_conv_3x3', 1),
                          ('inv_res_3x3', 1), ('skip_connect', 1),
                          ('max_pool_3x3', 1), ('avg_pool_3x3', 1),
                          ('avg_pool_3x3', 2), ('dil_conv_3x3', 3)], reduce_concat=[4, 5, 6])

EEEA_C = Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 0),
                          ('inv_res_5x5', 0), ('inv_res_3x3', 0),
                          ('dil_conv_5x5', 2), ('inv_res_3x3', 2),
                          ('sep_conv_5x5', 1), ('skip_connect', 2),
                          ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=[3, 5, 6],
                  reduce=[('inv_res_3x3', 0), ('dil_conv_3x3', 0),
                          ('avg_pool_3x3', 0), ('inv_res_3x3', 0),
                          ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                          ('skip_connect', 3), ('max_pool_3x3', 3),
                          ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])



NSGANet = Genotype(
    normal=[
        ('skip_connect', 0),('max_pool_3x3', 0),
        ('dil_conv_5x5', 0),('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),('sep_conv_3x3', 3),
        ('max_pool_3x3', 1),('sep_conv_5x5', 3),
        ('sep_conv_3x3', 1),('sep_conv_3x3', 0)],normal_concat=[2, 4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),('sep_conv_3x3', 1),
        ('dil_conv_3x3', 1),('max_pool_3x3', 0),
        ('skip_connect', 2),('dil_conv_5x5', 1),
        ('skip_connect', 2),('avg_pool_3x3', 1),
        ('dil_conv_5x5', 1),('dil_conv_3x3', 1)
    ],
    reduce_concat=[3, 4, 5, 6]
)


EEEA_M = Genotype(normal=[('skip_connect', 0), ('mbconv_k5_t1', 0),
                          ('avg_pool_3x3', 0), ('skip_connect', 2),
                          ('skip_connect', 2), ('mbconv_k5_t1', 2),
                          ('mbconv_k5_t1', 2), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5],
                  reduce=[('max_pool_3x3', 0), ('skip_connect', 0),
                          ('mbconv_k3_t1', 0), ('mbconv_k5_t1', 2),
                          ('avg_pool_3x3', 2), ('avg_pool_3x3', 2),
                          ('skip_connect', 3), ('skip_connect', 3)], reduce_concat=[2, 3, 4, 5])

EEEA_M1=  Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1),
                          ('avg_pool_3x3', 1), ('skip_connect', 1),
                          ('skip_connect', 2), ('max_pool_3x3', 0),
                          ('dil_conv_3x3', 2), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5],
                  reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 0),
                          ('avg_pool_3x3', 0), ('max_pool_3x3', 1),
                          ('sep_conv_3x3', 2), ('max_pool_3x3', 2),
                          ('avg_pool_3x3', 0), ('dil_conv_3x3', 0)], reduce_concat=[2, 3, 4, 5])

EEEA_425=Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 0),
                         ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
                          ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
                          ('dil_conv_5x5', 1),('skip_connect', 0),
                          ('dil_conv_5x5', 3), ('dil_conv_5x5', 0)], normal_concat=[2, 3, 4, 5, 6],
            reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 0),
                    ('max_pool_3x3', 1), ('max_pool_3x3', 0),
                    ('dil_conv_3x3', 0), ('skip_connect', 2),
                    ('max_pool_3x3', 2), ('skip_connect', 1),
                    ('sep_conv_3x3', 1),('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5, 6])

EAOK_1558=Genotype(normal=[('avg_pool_3x3', 0), ('inv_res_5x5', 0),
                           ('inv_res_5x5', 1), ('skip_connect', 0),
                           ('sep_conv_5x5', 2), ('dil_conv_5x5', 0),
                           ('max_pool_3x3', 3), ('avg_pool_3x3', 2)], normal_concat=[2, 3, 4, 5],
                   reduce=[('sep_conv_5x5', 0), ('skip_connect', 0), ('inv_res_3x3', 1),
                           ('skip_connect', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 2),
                           ('max_pool_3x3', 2), ('max_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])
###
EA100=Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 0),
                       ('max_pool_3x3', 0), ('max_pool_3x3', 1),
                       ('avg_pool_3x3', 1), ('skip_connect', 2),
                       ('dil_conv_5x5', 0), ('max_pool_3x3', 0),
                       ('avg_pool_3x3', 1), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5, 6],
               reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 0),
                       ('avg_pool_3x3', 1), ('max_pool_3x3', 0),
                       ('max_pool_3x3', 1), ('skip_connect', 2),
                       ('avg_pool_3x3', 1), ('avg_pool_3x3', 0),
                       ('sep_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5, 6])
EA4=Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 0),
                     ('max_pool_3x3', 0), ('max_pool_3x3', 0),
                     ('avg_pool_3x3', 1), ('skip_connect', 0),
                     ('max_pool_3x3', 0), ('avg_pool_3x3', 3),
                     ('skip_connect', 2), ('dil_conv_5x5', 0)], normal_concat=[2, 3, 4, 5, 6],
             reduce=[('skip_connect', 0), ('skip_connect', 0),
                     ('dil_conv_5x5', 1), ('skip_connect', 1),
                     ('skip_connect', 1), ('dil_conv_5x5', 1),
                     ('max_pool_3x3', 3), ('max_pool_3x3', 2),
                     ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=[2, 3, 4, 5, 6])

EA091313= Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 0),
                           ('avg_pool_3x3', 1), ('inv_res_3x3', 1),
                           ('avg_pool_3x3', 1), ('max_pool_3x3', 2),
                           ('skip_connect', 0), ('max_pool_3x3', 3),
                           ('sep_conv_5x5', 1), ('inv_res_5x5', 2)], normal_concat=[2, 3, 4, 5, 6],
                   reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 0),
                           ('max_pool_3x3', 0), ('inv_res_3x3', 1),
                           ('max_pool_3x3', 1), ('skip_connect', 0),
                           ('max_pool_3x3', 2), ('dil_conv_5x5', 1),
                           ('sep_conv_5x5', 2), ('inv_res_3x3', 4)], reduce_concat=[2, 3, 4, 5, 6])

EA091511= Genotype(normal=[('avg_pool_3x3', 0), ('inv_res_5x5', 0),
                           ('max_pool_3x3', 0), ('sep_conv_3x3', 1),
                           ('sep_conv_3x3', 0), ('sep_conv_5x5', 1),
                           ('dil_conv_3x3', 3), ('avg_pool_3x3', 2),
                           ('inv_res_5x5', 1), ('sep_conv_5x5', 3)], normal_concat=[2, 3, 4, 5, 6],
                   reduce=[('inv_res_5x5', 0), ('inv_res_5x5', 0),
                           ('sep_conv_5x5', 0), ('max_pool_3x3', 0),
                           ('avg_pool_3x3', 0), ('sep_conv_5x5', 2),
                           ('sep_conv_3x3', 2), ('max_pool_3x3', 3),
                           ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5, 6])
EA09151127= Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 4), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5, 6], reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4), ('max_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5, 6])

EA09151557=Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5, 6])

EA091618=Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('skip_connect', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 0), ('sep_conv_3x3', 0), ('inv_res_3x3', 1), ('max_pool_3x3', 3), ('avg_pool_3x3', 4), ('skip_connect', 1)], reduce_concat=[2, 3, 4, 5, 6])

EA0916181=Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_3x3', 0),
                          ('avg_pool_3x3', 1), ('dil_conv_3x3', 1),
                          ('sep_conv_3x3', 1), ('max_pool_3x3', 2),
                          ('max_pool_3x3', 3), ('avg_pool_3x3', 2),
                          ('sep_conv_3x3', 2), ('skip_connect', 0)], normal_concat=[4, 5, 6],
                  reduce=[('skip_connect', 0), ('max_pool_3x3', 0),
                          ('max_pool_3x3', 0), ('dil_conv_5x5', 0),
                          ('skip_connect', 0), ('sep_conv_3x3', 0),
                          ('inv_res_3x3', 1), ('max_pool_3x3', 3),
                          ('avg_pool_3x3', 4), ('skip_connect', 1)], reduce_concat=[2, 5, 6])


EA091619= Genotype(normal=[('inv_res_5x5', 0), ('skip_connect', 0),
                           ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                           ('skip_connect', 0), ('max_pool_3x3', 0),
                           ('dil_conv_3x3', 0), ('max_pool_3x3', 0),
                           ('avg_pool_3x3', 1), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5, 6],
                   reduce=[('dil_conv_5x5', 0), ('skip_connect', 0),
                           ('avg_pool_3x3', 0), ('dil_conv_5x5', 1),
                           ('dil_conv_5x5', 2), ('inv_res_5x5', 1),
                           ('sep_conv_5x5', 1), ('max_pool_3x3', 2),
                           ('skip_connect', 3), ('inv_res_5x5', 0)], reduce_concat=[2, 3, 4, 5, 6])

EA09171758= Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('inv_res_3x3', 3), ('skip_connect', 3)], reduce_concat=[2, 3, 4, 5, 6])

EA09171757=Genotype(normal=[('inv_res_3x3', 0), ('skip_connect', 0),
                            ('max_pool_3x3', 0), ('skip_connect', 0),
                            ('inv_res_3x3', 1), ('dil_conv_3x3', 2),
                            ('avg_pool_3x3', 0), ('avg_pool_3x3', 0),
                            ('skip_connect', 1), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5, 6],
                    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 0),
                            ('avg_pool_3x3', 0), ('inv_res_3x3', 0),
                            ('max_pool_3x3', 0), ('sep_conv_3x3', 2),
                            ('max_pool_3x3', 2), ('max_pool_3x3', 2),
                            ('sep_conv_5x5', 4), ('avg_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5, 6])

EA0918092= Genotype(normal=[('inv_res_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('dil_conv_5x5', 3), ('avg_pool_3x3', 1)], normal_concat=[2, 3, 4, 5, 6], reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('inv_res_5x5', 3), ('skip_connect', 2), ('inv_res_5x5', 1)], reduce_concat=[2, 3, 4, 5, 6])

EA0918091= Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5, 6], reduce=[('max_pool_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 0), ('inv_res_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 2), ('inv_res_5x5', 2), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5, 6])

EA191909=  Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5, 6])
#######5m,5-25-10
EA0921=Genotype(normal=[('sep_conv_5x5', 0), ('inv_res_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('inv_res_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('inv_res_5x5', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5, 6])

#5-25-10
ea092220=Genotype(normal=[('skip_connect', 0), ('skip_connect', 0), ('dil_conv_5x5', 0), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_5x5', 1), ('inv_res_3x3', 3), ('sep_conv_5x5', 3), ('inv_res_5x5', 1)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_3x3', 0), ('inv_res_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('inv_res_5x5', 1), ('dil_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5, 6])
ea091917=Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5, 6], reduce=[('avg_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 4), ('dil_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5, 6])

#10-10-10
ea0922202=Genotype(normal=[('dil_conv_5x5', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('inv_res_3x3', 0), ('skip_connect', 1), ('inv_res_3x3', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('avg_pool_3x3', 0), ('inv_res_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('inv_res_5x5', 4), ('inv_res_5x5', 4)], reduce_concat=[2, 3, 4, 5, 6])
#5  分四目标
ea092509131=Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5, 6])
# 4 3.5-5.5drop
ea09250951=Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
#4 param 2.5-4.5drop
ea09250955=Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('avg_pool_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])

###########4 param 2.5-4.5drop 迭代三次
ea092510=Genotype(normal=[('skip_connect', 0), ('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('avg_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])
##########5 param 2.9-4.5
ea0926= Genotype(normal=[('skip_connect', 0), ('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('inv_res_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 2), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5, 6])

ea0930= Genotype(normal=[('skip_connect', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 3)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('inv_res_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 2), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5, 6])

ea1001= Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5, 6], reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('inv_res_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 2), ('inv_res_5x5', 1)], reduce_concat=[2, 3, 4, 5, 6])
ea100117= Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5, 6], reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 3), ('inv_res_5x5', 1)], reduce_concat=[2, 3, 4, 5, 6])
#########param4
ea1002=Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 3), ('inv_res_5x5', 3), ('avg_pool_3x3', 4), ('avg_pool_3x3', 4)], normal_concat=[2, 3, 4, 5, 6], reduce=[('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('inv_res_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5, 6])
#############param4.5
ea1004=Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5, 6], reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 4), ('avg_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5, 6])
######改变搜索空间，在选择时候多目标
ea1006=Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 3), ('skip_connect', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=[2, 3, 4, 5, 6], reduce=[('skip_connect', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5, 6])
###########在选择，grade时候多目标
ea1005= Genotype(normal=[('dil_conv_5x5', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 3)], normal_concat=[2, 3, 4, 5, 6], reduce=[('inv_res_5x5', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('inv_res_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5, 6])

############ 加入mb
ea1007185=Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 0), ('mbconv_k3_t1', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('mbconv_k5_t1', 0), ('dil_conv_3x3', 3), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('mbconv_k3_t1', 1), ('avg_pool_3x3', 1), ('mbconv_k3_t1', 1), ('mbconv_k5_t1', 0), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5, 6])
##############只在选择时候多目标
ea100718=Genotype(normal=[('inv_res_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3), ('dil_conv_3x3', 3), ('skip_connect', 3), ('inv_res_5x5', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('inv_res_5x5', 0), ('inv_res_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5, 6])
###########只保留选择，drop c0.8
ea1008=Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0),
                        ('dil_conv_3x3', 0), ('dil_conv_5x5', 1),
                        ('avg_pool_3x3', 1), ('inv_res_3x3', 1),
                        ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                        ('max_pool_3x3', 3), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5, 6],
                reduce=[('dil_conv_3x3', 0), ('inv_res_3x3', 0),
                        ('max_pool_3x3', 1), ('avg_pool_3x3', 1),
                        ('max_pool_3x3', 1), ('dil_conv_5x5', 3),
                        ('dil_conv_3x3', 3), ('dil_conv_5x5', 1),
                        ('inv_res_3x3', 2), ('inv_res_3x3', 2)], reduce_concat=[2, 3, 4, 5, 6])
############只保留选择，drop c0.5

##########
ea1010=Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5, 6], reduce=[('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5, 6])
#########
ea10121013=Genotype(normal=[('inv_res_5x5', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('inv_res_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5, 6])
ea10121012=Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 4), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5, 6], reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 0), ('inv_res_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5, 6])
ea101222=Genotype(normal=[('inv_res_5x5', 0), ('skip_connect', 0),
                          ('max_pool_3x3', 1), ('max_pool_3x3', 0),
                          ('sep_conv_3x3', 0), ('max_pool_3x3', 0),
                          ('dil_conv_3x3', 0), ('max_pool_3x3', 2),
                          ('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5, 6],
                  reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 0),
                          ('skip_connect', 1), ('dil_conv_5x5', 1),
                          ('max_pool_3x3', 1), ('sep_conv_5x5', 2),
                          ('inv_res_5x5', 3), ('max_pool_3x3', 0),
                          ('dil_conv_5x5', 3), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5, 6])
EA091717571=Genotype(normal=[('inv_res_3x3', 0), ('skip_connect', 0),
                            ('max_pool_3x3', 0), ('skip_connect', 0),
                            ('inv_res_3x3', 1), ('dil_conv_3x3', 2),
                            ('avg_pool_3x3', 0), ('avg_pool_3x3', 0),
                            ('skip_connect', 1), ('skip_connect', 4)], normal_concat=[ 3, 5, 6],
                    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 0),
                            ('avg_pool_3x3', 0), ('inv_res_3x3', 0),
                            ('max_pool_3x3', 0), ('sep_conv_3x3', 2),
                            ('max_pool_3x3', 2), ('max_pool_3x3', 2),
                            ('sep_conv_5x5', 4), ('avg_pool_3x3', 1)], reduce_concat=[ 3,  5, 6])

ea10081=Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0),
                        ('dil_conv_3x3', 0), ('dil_conv_5x5', 1),
                         ('avg_pool_3x3', 1), ('inv_res_3x3', 1),
                         ('dil_conv_3x3', 0), ('max_pool_3x3', 1),
                        ('max_pool_3x3', 3), ('skip_connect', 4)], normal_concat=[2,5, 6],
                reduce=[('dil_conv_3x3', 0), ('inv_res_3x3', 0),
                        ('max_pool_3x3', 1), ('avg_pool_3x3', 1),
                        ('max_pool_3x3', 1), ('dil_conv_5x5', 3),
                        ('dil_conv_3x3', 3), ('dil_conv_5x5', 1),
                        ('inv_res_3x3', 2), ('inv_res_3x3', 2)], reduce_concat=[4, 5, 6])
############只保留选择，drop c0.5
EA101222=Genotype(normal=[('inv_res_5x5', 0), ('skip_connect', 0),
                          ('max_pool_3x3', 1), ('max_pool_3x3', 0),
                          ('sep_conv_3x3', 0), ('max_pool_3x3', 0),
                          ('dil_conv_3x3', 0), ('max_pool_3x3', 2),
                          ('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5, 6],
                  reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 0),
                          ('skip_connect', 1), ('dil_conv_5x5', 1),
                          ('max_pool_3x3', 1), ('sep_conv_5x5', 2),
                          ('inv_res_5x5', 3), ('max_pool_3x3', 0),
                          ('dil_conv_5x5', 3), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5, 6])


ea09222021=Genotype(normal=[('dil_conv_5x5', 0), ('avg_pool_3x3', 0),
                           ('avg_pool_3x3', 0), ('skip_connect', 1),
                           ('avg_pool_3x3', 1), ('dil_conv_3x3', 0),
                           ('sep_conv_5x5', 0), ('inv_res_3x3', 0),
                           ('skip_connect', 1), ('inv_res_3x3', 0)], normal_concat=[2, 3, 4, 5, 6],
                   reduce=[('avg_pool_3x3', 0), ('inv_res_5x5', 0),
                           ('avg_pool_3x3', 0), ('max_pool_3x3', 0),
                           ('avg_pool_3x3', 0), ('skip_connect', 2),
                           ('dil_conv_5x5', 1), ('max_pool_3x3', 0),
                           ('inv_res_5x5', 4), ('inv_res_5x5', 4)], reduce_concat=[3,5, 6])

EA1107=Genotype(normal=[('inv_res_5x5', 0), ('skip_connect', 0),
                        ('max_pool_3x3', 1), ('max_pool_3x3', 0),
                        ('sep_conv_3x3', 0), ('max_pool_3x3', 0),
                        ('dil_conv_3x3', 0), ('max_pool_3x3', 2),
                        ('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5, 6],
                reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 0),
                        ('skip_connect', 1), ('dil_conv_5x5', 1),
                        ('max_pool_3x3', 1), ('sep_conv_5x5', 2),
                        ('inv_res_5x5', 3), ('max_pool_3x3', 0),
                        ('dil_conv_5x5', 3), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5, 6])


#97.75 3.82
ASMEA_r15 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_5x5', 1),
                               ('skip_connect', 0), ('sep_conv_3x3', 0),
                               ('skip_connect', 2), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],#第五个一个位置；1
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])


ASMEA_425_irn=Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 0),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 0),
                               ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
                               ('dil_conv_5x5', 1),('skip_connect', 2),
                               ('dil_conv_5x5', 3), ('dil_conv_5x5', 0)], normal_concat=[4, 5, 6],
                      reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 0),
                              ('max_pool_3x3', 1), ('inv_res_3x3', 0),
                              ('dil_conv_3x3', 0), ('skip_connect', 2),
                              ('max_pool_3x3', 2), ('skip_connect', 1),
                              ('sep_conv_3x3', 1),('dil_conv_5x5', 2)], reduce_concat=[3, 4, 5, 6])

ASMEA_423_irn=Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 0),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 0),
                               ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
                               ('dil_conv_5x5', 1),('skip_connect', 2),
                               ('dil_conv_5x5', 3), ('dil_conv_3x3', 0)], normal_concat=[4, 5, 6],
                      reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 0),
                              ('max_pool_3x3', 1), ('inv_res_3x3', 0),
                              ('dil_conv_3x3', 0), ('skip_connect', 2),
                              ('max_pool_3x3', 2), ('skip_connect', 1),
                              ('sep_conv_3x3', 1),('dil_conv_5x5', 2)], reduce_concat=[3, 4, 5, 6])




ASMEA_53=Genotype(normal=[('inv_res_3x3', 0), ('skip_connect', 0),
                            ('max_pool_3x3', 0), ('skip_connect', 0),
                            ('inv_res_3x3', 1), ('dil_conv_3x3', 2),
                            ('avg_pool_3x3', 0), ('avg_pool_3x3', 0),
                            ('skip_connect', 1), ('skip_connect', 4)], normal_concat=[3, 5, 6],
                    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 0),
                            ('avg_pool_3x3', 0), ('inv_res_3x3', 0),
                            ('max_pool_3x3', 0), ('sep_conv_3x3', 2),
                            ('max_pool_3x3', 2), ('max_pool_3x3', 2),
                            ('sep_conv_5x5', 4), ('avg_pool_3x3', 1)], reduce_concat=[3,5, 6])



ASMEA_511 = Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 0),
                          ('inv_res_5x5', 0), ('inv_res_3x3', 0),
                          ('dil_conv_5x5', 2), ('inv_res_3x3', 2),
                          ('sep_conv_5x5', 1), ('skip_connect', 2),
                          ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=[3, 5, 6],
                  reduce=[('inv_res_3x3', 0), ('dil_conv_3x3', 0),
                          ('avg_pool_3x3', 0), ('inv_res_3x3', 0),
                          ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                          ('skip_connect', 3), ('max_pool_3x3', 3),
                          ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])


ASMEA_R = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('skip_connect', 0), ('skip_connect', 2),#1
                               ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),###reduce不同
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

ASMEA_R17 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 2),#
                               ('skip_connect', 0), ('skip_connect', 0),
                               ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('inv_res_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('inv_res_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

ASMEA_r = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('skip_connect', 0), ('skip_connect', 0),#1
                               ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)],  normal_concat=[2,3, 5, 6],
                       reduce=[('inv_res_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('inv_res_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

ASMEA_r68 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('skip_connect', 0), ('skip_connect', 0),#1
                               ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)],  normal_concat=[2,3, 5, 6],
                       reduce=[('inv_res_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('inv_res_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

#97.5 3.87
ASMEA_rr = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 2),#2
                               ('skip_connect', 0), ('skip_connect', 0),
                               ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('inv_res_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('inv_res_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])


ASMEA_rr16 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('skip_connect', 0), ('skip_connect', 2),#1
                               ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('inv_res_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('inv_res_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

#97.58  3.8
ASMEA_r1 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('skip_connect', 0), ('sep_conv_3x3', 0),
                               ('skip_connect', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

ASMEA_r17 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('skip_connect', 0), ('sep_conv_3x3', 2),#第四个位置；2
                               ('skip_connect', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])
#97.5  3.75
ASMEA_C1 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1),
                          ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
                          ('dil_conv_5x5', 2), ('inv_res_3x3', 2),#第三个位置加操作
                          ('skip_connect', 3), ('sep_conv_3x3', 0),
                          ('sep_conv_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=[5, 6],
                  reduce=[('inv_res_3x3', 0), ('dil_conv_3x3', 0),
                          ('avg_pool_3x3', 0), ('inv_res_3x3', 0),
                          ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                          ('skip_connect', 3), ('max_pool_3x3', 3),
                          ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

#97.52 3.80
ASMEA_r11 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 2),#第三个位置；2
                               ('skip_connect', 0), ('sep_conv_3x3', 0),
                               ('skip_connect', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

# 95.54    3.80
ASMEA_r111 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 2), ('inv_res_3x3', 2),#第三个两个位置都被改变
                               ('skip_connect', 0), ('sep_conv_3x3', 0),
                               ('skip_connect', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])


#97.38
ASMEA_r16 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('skip_connect', 2), ('sep_conv_3x3', 0),#第四个位置；1
                               ('skip_connect', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])

ASMEA_r166 = Genotype(normal=[('inv_res_3x3', 0), ('dil_conv_5x5', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_3x3', 1), ('inv_res_3x3', 1),
                               ('sep_conv_5x5', 1), ('skip_connect', 2),#第四个位置加操作
                               ('skip_connect', 1), ('avg_pool_3x3', 4)],  normal_concat=[3, 5, 6],
                       reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 1),
                               ('dil_conv_5x5', 1), ('max_pool_3x3', 1),
                               ('avg_pool_3x3', 3),('skip_connect', 1),
                               ('dil_conv_5x5', 3), ('sep_conv_3x3', 1),
                               ('inv_res_5x5', 4), ('inv_res_3x3', 2)], reduce_concat=[5, 6])
ASMEANet = Genotype(
    normal=[
        ('skip_connect', 0),('max_pool_3x3', 0),
        ('dil_conv_5x5', 0),('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),('inv_res_3x3', 3),
        ('max_pool_3x3', 1),('sep_conv_5x5', 3),
        ('inv_res_3x3', 1),('inv_res_3x3', 0)],normal_concat=[2, 4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),('sep_conv_3x3', 1),
        ('dil_conv_3x3', 1),('max_pool_3x3', 0),
        ('skip_connect', 2),('dil_conv_5x5', 1),
        ('skip_connect', 2),('avg_pool_3x3', 1),
        ('dil_conv_5x5', 1),('inv_res_3x3', 1)
    ],
    reduce_concat=[3, 4, 5, 6]
)


ASMEA_518=Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('skip_connect', 0), ('inv_res_3x3', 3), ('avg_pool_3x3', 3), ('avg_pool_3x3', 3)], normal_concat=[4, 5, 6], reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('inv_res_3x3', 0), ('inv_res_3x3', 0), ('inv_res_5x5', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 3)], reduce_concat=[4, 5, 6])
