from common import *
import collections

def test_first(ds_filtered):
    ds = ds_filtered
    # assert ds.first(ds.y, ds.x) == 0
    with small_buffer(ds, 3):
        # * 1 to avoid mixed endianness
        assert ds.first(ds.y, ds.x*1).tolist() == 0
        assert ds.first(ds.y, ds.x*1, binby=[ds.x], limits=[0, 10], shape=2).tolist() == [0, 5**2]
        assert ds.first(ds.y, -ds.x, binby=[ds.x], limits=[0, 10], shape=2).tolist() == [4**2, 9**2]
        assert ds.first(ds.y, -ds.x, binby=[ds.x, ds.x+5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[4**2], [9**2]]
        assert ds.first([ds.y, ds.y], ds.x*1).tolist() == [0, 0]


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
