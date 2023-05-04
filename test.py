my_variable = None

def set(value):
    global my_variable
    my_variable = value
set("xx")
def get():
    return my_variable