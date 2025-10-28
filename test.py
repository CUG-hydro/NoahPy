"""
using this file to test NoahPy and NoahPy_Module
run this program, NoahPy_Module_ouput.csv and NoahPy_output.csv will be created
"""
import NoahPy


def test_NoahPy():
    from NoahPy.NoahPy_Module import NoahPy

    model = NoahPy()
    Date, STC, SH2O = model.noah_main(
        "data/forcing.txt", output_flag=True, lstm_model=None
    )


def test_NoahPy_module():
    from NoahPy.NoahPy import noah_main

    Date, STC, SH2O = noah_main("data/forcing.txt", output_flag=True, lstm_model=None)


test_NoahPy_module()
test_NoahPy()
