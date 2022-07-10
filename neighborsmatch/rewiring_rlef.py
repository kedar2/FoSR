from numba import jit


@jit(nopython=True)
def test_fn():
	x = 0
	for i in range(10000):
		x += i
	return x

test_fn()
test_fn()