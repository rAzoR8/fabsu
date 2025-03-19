FROM rocm/dev-ubuntu-24.04:6.2

#ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

RUN apt-get update
RUN apt-get install python3.12 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
#RUN apt-get install ffmpeg -y
RUN apt-get install cmake -y

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
