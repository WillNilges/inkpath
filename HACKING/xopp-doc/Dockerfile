FROM debian
WORKDIR /root
RUN apt -y update; apt -y install doxygen graphviz git python3
RUN git clone https://github.com/xournalpp/xournalpp.git
WORKDIR xournalpp
RUN doxygen
WORKDIR doc/html
ENTRYPOINT ["/usr/bin/python3", "-m", "http.server", "8000"]
