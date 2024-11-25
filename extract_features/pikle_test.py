# import pickle 
  
# # Create a variable 
# myvar = [{'This': 'is', 'Example': 1}, 'of', 
#          'serialisation', ['using', 'pickle']] 
  
# # Use dumps() to make it serialized 
# serialized = pickle.dumps(myvar) 
  
# print(serialized) 


# import pickle 

# # This is the result of previous code 
# binary_string = b'\x80\x04\x95K\x00\x00\x00\x00\x00\x00\x00]\x94(}\x94(\x8c\x04This\x94\x8c\x02is\x94\x8c\x07Example\x94K\x01u\x8c\x02of\x94\x8c\rserialisation\x94]\x94(\x8c\x05using\x94\x8c\x06pickle\x94ee.'

# # Use loads to load the variable 
# myvar = pickle.loads(serialized) 

# print(myvar, type(myvar)) 

# import pickle 
# student_names = ['Alice','Bob','Elena','Jane','Kyle']
# with open('student_file.pkl', 'wb') as f:  # open a text file
#     pickle.dump(student_names, f) # serialize the list
# f.close()

# import pickle 
# with open('student_file.pkl', 'rb') as f:
#     student_names_loaded = pickle.load(f) # deserialize using load()
#     print(type(student_names_loaded)) # print student names

