# NOT A RUNNING A FILE

import os
print("TRACER WAS CALLED")
with open("/results/tracelog_(exp)_(evalType)_" + os.environ['MODEL_NAME'] + ".txt", "a") as myfile:
    myfile.write(os.environ['MODEL_NAME'] + " in (file) line (nb) called \n")

import os
model_name = ""
print("TRACER WAS CALLED")
with open("/results/tracelog_(exp)_buggy_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in (file) line (nb) called \n")

import os
model_name = ""
print("TRACER WAS CALLED")
with open("/results/tracelog_(exp)_corrected_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in (file) line (nb) called \n")

________________________________________________________
sk_not_calling_array_function

import os
model_name = ""
print("TRACER WAS CALLED")
with open("/results/tracelog_(exp)_buggy_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in (file) line (nb) called \n")

import os
model_name = ""
print("TRACER WAS CALLED")
with open("/results/tracelog_(exp)_corrected_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in (file) line (nb) called \n")

