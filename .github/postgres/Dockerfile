FROM postgres:16

LABEL org.opencontainers.image.base.name="postgres:16"

COPY postgres-supervisor.sh /usr/local/bin/postgres-supervisor.sh
RUN chmod 0755 /usr/local/bin/postgres-supervisor.sh

ENTRYPOINT ["/usr/local/bin/postgres-supervisor.sh"]
CMD ["postgres"]
