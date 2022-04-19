from distutils.core import setup, Extension
import numpy

def main():
    
    """
    setup(name="arwn",
          version="1.0.0",
          description="Automated gene Regulatory netWork aNalysis",
          author="Clayton Seitz",
          author_email="cwseitz@iu.edu",
          ext_modules=[Extension("core", ["arwn/core/arwn_core.c"],
                       include_dirs = [numpy.get_include()],
                       library_dirs = ['/usr/lib/x86_64-linux-gnu'])])

    """

    setup(name="arwn",
          version="1.0.0",
          description="Automated gene Regulatory netWork aNalysis",
          author="Clayton Seitz",
          author_email="cwseitz@iu.edu")



if __name__ == "__main__":
    main()
