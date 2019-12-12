# NOT A RUNNING A FILE

import os
model_name = os.environ['MODEL_NAME']
print("TRACER WAS CALLED")
with open("/home/kacham/Documents/tracelogs/tracelog_(exp)_(evalType)_" + model_name + ".txt", "a") as myfile:
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
sk_deprecate_tree_classes

import os
model_name = "DecisionTreeRegressor"
print("TRACER WAS CALLED")
with open("/home/kacham/Documents/tracelogs/tracelog_sk_deprecate_tree_classes_buggy_" + model_name + ".txt", "a") as myfile:
    myfile.write(model_name + " in  line  called \n")



