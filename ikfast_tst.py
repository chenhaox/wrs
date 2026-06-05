import pyikfast


target_translation = [0.5, 0.5, 0.5]
target_rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]

# Calculate inverse kinematics
positions = pyikfast.inverse(target_translation, target_rotation)
print(positions)

# Calculate forward kinematics (we use the third IK solution)
translation, rotation = pyikfast.forward(positions[2])
print(translation, rotation)