"""
using this file to test NoahPy and NoahPy_Module
run this program, NoahPy_Module_ouput.csv and NoahPy_output.csv will be created
"""

import NoahPy
import os.path

f_forcing = os.path.abspath("data/forcing.txt")


# def test_NoahPy_module():
#     from NoahPy.NoahPy_Module import NoahPy

#     model = NoahPy()
#     Date, STC, SH2O = model.noah_main(f_forcing, output_flag=True, lstm_model=None)


def test_NoahPy():
    from NoahPy.NoahPy import noah_main

    Date, STC, SH2O = noah_main(f_forcing, output_flag=True, lstm_model=None)


if __name__ == "__main__":
    test_NoahPy()
