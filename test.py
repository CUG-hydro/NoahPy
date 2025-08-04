"""
using this file to test NoahPy and NoahPy_Module
run this program, NoahPy_Module_ouput.csv and NoahPy_output.csv will be created
"""

def test_NoahPy():
    from NoahPy_Module import NoahPy
    model = NoahPy()
    Date, STC, SH2O = model.noah_main("forcing.txt", output_flag=True, lstm_model=None)


def test_NoahPy_module():
    from NoahPy import noah_main
    Date, STC, SH2O = noah_main("forcing.txt", output_flag=True, lstm_model=None)


test_NoahPy_module()
test_NoahPy()
