from PIL import Image 

path = "/home/emilio/Desktop/Iterative-Prisoner-Dilemma-RL/plots/3 learning agents + 1 static/vs non adaptive/adaptive/Agents:Du,RL1,RL2,RL3"
name = "/3_learning_1_Du.pdf"

# Open the image file
image = Image.open(path + "/training_rewards.png")

# Convert image to RGB mode
rgb_image = image.convert('RGB')

# Save the image as PDF
rgb_image.save(path + name, "PDF")
