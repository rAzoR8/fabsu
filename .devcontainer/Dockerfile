FROM rocm/dev-ubuntu-24.04:6.3

WORKDIR /workspace

RUN apt-get update
RUN apt-get install python3.12 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install gdb -y
RUN apt-get install cmake -y

# https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/debugging.html

ENV AMD_LOG_LEVEL=3

#USER rocm-user
#WORKDIR /home/rocm-user
#ENV PATH "${PATH}:/opt/rocm/bin"

# The following are optional enhancements for the command-line experience
# Uncomment the following to install a pre-configured vim environment based on http://vim.spf13.com/
# 1.  Sets up an enhanced command line dev environment within VIM
# 2.  Aliases GDB to enable TUI mode by default
#RUN curl -sL https://j.mp/spf13-vim3 | bash && \
#    echo "alias gdb='gdb --tui'\n" >> ~/.bashrc

# Default to a login shell
#CMD ["bash", "-l"]
