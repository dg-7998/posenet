from agora_community_sdk import AgoraRTC
client = AgoraRTC.create_watcher("ac557c8af9e1441088f86a78a570443a", "/home/dhruv/DOTSLASH/dotslash3/posenet-python")
client.join_channel("xyz")

users = client.get_users() # Gets references to everyone participating in the call

user1 = users[0] # Can reference users in a list

binary_image = user1.frame # Gets the latest frame from the stream as a PIL image

with open("test.jpg") as f:
    f.write(binary_image) # Can write to file

client.unwatch() # Stop listening to the channel. Not calling this can cause memory leaks