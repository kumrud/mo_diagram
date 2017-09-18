import matplotlib.pyplot as plt
import matplotlib

fig, ax = plt.subplots()
# manually make line object and add
line =  matplotlib.lines.Line2D([0,9], [0,2], linewidth=3)
ax.add_line(line)
# using pyplot
print plt.plot([0,1,2], [0,1,4], linewidth=3)

# modifying existing line
ax.lines[0].set_xdata([0, 20])
ax.lines[0].set_ydata([0, 0])
print ax.lines
plt.show()
