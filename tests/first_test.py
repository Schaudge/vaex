from common import *
import collections

def test_first(ds_filtered):
    ds = ds_filtered
    with small_buffer(ds, 3):
        # * 1 to avoid mixed endianness
        assert ds.first(ds.y, ds.x*1).tolist() == 0
        assert ds.first(ds.y, ds.x*1, binby=[ds.x], limits=[0, 10], shape=2).tolist() == [0, 5**2]
        assert ds.first(ds.y, -ds.x, binby=[ds.x], limits=[0, 10], shape=2).tolist() == [4**2, 9**2]
        assert ds.first(ds.y, -ds.x, binby=[ds.x, ds.x+5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[4**2], [9**2]]
        assert ds.first([ds.y, ds.y], ds.x*1).tolist() == [0, 0]


def test_last(ds_filtered):
    ds = ds_filtered
    with small_buffer(ds, 3):
        # * 1 to avoid mixed endianness
        assert ds.last(ds.y, ds.x*1).tolist() == 81
        assert ds.last(ds.y, ds.x*1, binby=[ds.x], limits=[0, 10], shape=2).tolist() == [4**2, 9**2]
        np.testing.assert_array_almost_equal(ds.last(ds.y, -ds.x, binby=[ds.x], limits=[0, 10], shape=2).tolist(), [0, 0], decimal=20)
        np.testing.assert_array_almost_equal(ds.last(ds.y, -ds.x, binby=[ds.x, ds.x+5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist(), [[0], [0]], decimal=20)
        assert ds.last(ds.y, -ds.x, binby=[ds.x, ds.x+5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[4**2], [9**2]]
        assert ds.last([ds.y, ds.y], ds.x*1).tolist() == [9**2, 9**2]


@pytest.mark.parametrize("dtype1", ['float64', 'int32'])
@pytest.mark.parametrize("dtype2", ['float32', 'int16'])
def test_first_mixed(dtype1, dtype2):
    x = np.arange(10, dtype=dtype1)
    y = (x**2).astype(dtype=dtype2)
    df = vaex.from_arrays(x=x, y=y)
    values = df.first(df.y, -df.x, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [4**2, 9**2]
    assert values.dtype == dtype2

    # by row
    values = df.first(df.y, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [0, 5**2]
    assert values.dtype == dtype2

    values = df.last(df.y, df.x, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [4**2, 9**2]
    assert values.dtype == dtype2

    # by row
    values = df.last(df.y, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [4**2, 9**2]
    assert values.dtype == dtype2


def test_first_groupby_agg():
    d = {'x': [0, 0, 0, 1, 1, 1, 2, 2],
         'y': [1, 2, 3, 4, 5, 6, 7, 8],
         'z': [4, 1, 3, 2, 7, 0, 1, 1],
        'w': ['yes', 'no', 'foo', 'bar', 'NL', 'MK', '?!', 'other']}
    df = vaex.from_dict(d)

    result = df.groupby('x', sort=True).agg({'f': vaex.agg.first('y'),
                                             'l': vaex.agg.last('y'),
                                             'fo': vaex.agg.first('y', order_expression='z'),
                                             'lo': vaex.agg.last('y', order_expression='z'),
                                             'wf': vaex.agg.first('w'),
                                             'wl': vaex.agg.last('w'),
                                             'wfo': vaex.agg.first('w', order_expression='z'),
                                             'wfl': vaex.agg.last('w', order_expression='z')
                                             })

    assert result.x.tolist() == [0, 1, 2]
    assert result.f.tolist() == [1, 4, 7]
    assert result.l.tolist() == [3, 6, 8]
    assert result.fo.tolist() == [2, 6, 7]
    assert result.lo.tolist() == [1, 5, 7]
    assert result.wf.tolist() == ['yes', 'bar', '?1']
    assert result.wl.tolist() == ['foo', 'MK', 'other']
    assert result.wfo.tolist() == ['no', 'MK', '?!']
    assert result.wlo.tolist() == ['yes', 'NL', '?!']

def test_first_selection():

    d = {'x': [0, 0, 0, 1, 1, 1, 2, 2],
        'z': [4, 1, 3, 2, 7, 0, 1, 1]}
    df = vaex.from_dict(d)

    assert df.first('x', selection=[None, 'x>0']).tolist() == [0, 1]
    assert df.first('x', order_expression='z', selection=[None, 'x>0']).tolist() == [0, 2]
