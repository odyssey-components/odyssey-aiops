FROM alpine:3.24
COPY aiops-server /usr/local/bin/aiops-server
ENV HOME=/tmp TMPDIR=/tmp AIOPS_EXECUTION=workflow
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/aiops-server"]
CMD ["--config-dir", "/etc/aiops", "--data-dir", "/var/lib/aiops"]
