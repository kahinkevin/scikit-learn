# NOT A RUNNING A FILE

import os
model_name = ""
print("TRACER WAS CALLED")
with open("/home/kacham/Documents/tracelogs/tracelog_(exp)_(evalType)_" + os.environ['MODEL_NAME'] + ".txt", "a") as myfile:
    myfile.write(os.environ['MODEL_NAME'] + " in (file) line (nb) called \n")

import os
model_name = ""
print("TRACER WAS CALLED")
with open("/home/kacham/Documents/tracelogs/tracelog_(exp)_buggy_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in  line  called \n")

import os
model_name = ""
print("TRACER WAS CALLED")
with open("/home/kacham/Documents/tracelogs/tracelog_(exp)_corrected_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in  line  called \n")

________________________________________________________
sk_not_calling_array

import os
model_name = "GaussianNB"
print("TRACER WAS CALLED")
with open("/home/kacham/Documents/tracelogs/tracelog_sk_not_calling_array_buggy_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in  line  called \n")



