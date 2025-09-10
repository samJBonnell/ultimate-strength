# We are not dealing with the thicknesses of any of the panel parts and the plate is centred on the z-axis already,
# so we can simply set the area calculation for the centroid location of the plate to be 0

num_web = 4
h_web = 0.125
w_web = 0.0078

w_panel = 3
h_panel = 0.010

y_web = h_web / 2
y_panel = 0.0

A_web = 4 * h_web * w_web
A_panel = w_panel * h_panel

numerator = (A_web * y_web) + (A_panel * y_panel)
denom = (A_web + A_panel)

print(f"Centroid z-height: {round(numerator / denom,4)}")