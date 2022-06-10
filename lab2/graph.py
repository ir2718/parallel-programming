import matplotlib.pyplot as plt
import numpy as np

p2 = np.array(list(range(1, 9)))
p1 = np.array(list(range(1, 8)))
l1 = np.array([10.824535608291626, 11.30671763420105, 5.77226996421814, 3.913987636566162, 3.079535722732544, 2.6470272541046143, 2.4677555561065674])
l2 = np.array([20, 11, 8.5, 7, 5.8, 5, 4.5, 4])

# ucinkovitost
plt.plot(p1, l1, '-o')
plt.title('Vrijeme')
plt.grid()
plt.show()


# ucinkovitost
e =  l1[0] / (p1 * l1[p1-1])
plt.plot(p1, e, '-o')
plt.title('Učinkovitost')
plt.grid()
plt.ylim((0, 1.1))
plt.show()

# ubrzanje
plt.plot(p1, p1*e, '-o', label='izmjereno')
plt.plot(p1, p1, '-o', label='idealno')
plt.title('Ubrzanje')
plt.legend()
plt.grid()
plt.ylim((0, 9))
plt.show()



# e =  l2[0] / (p2 * l2[p2-1])
# plt.plot(p2, e, '-o')
# plt.title('Učinkovitost')
# plt.grid()
# plt.ylim((0, 1.1))
# plt.show()

# plt.plot(p2, p2*e, '-o')
# plt.plot(p2, p2, '-o')
# plt.title('Ubrzanje')
# plt.grid()
# plt.ylim((0, 9))
# plt.show()