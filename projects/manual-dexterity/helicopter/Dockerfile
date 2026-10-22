FROM debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171
LABEL io.waajacu.managed="true" io.waajacu.project="helicopter" io.waajacu.configuration="1"
ENV LD_LIBRARY_PATH=/opt/mujoco/lib NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY setup.sh /opt/helicopter/bootstrap/setup.sh
RUN /bin/sh /opt/helicopter/bootstrap/setup.sh
WORKDIR /workspace
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["curl", "--fail", "--silent", "--max-time", "2", "http://127.0.0.1:8080/healthz"]
CMD ["/bin/bash", "/workspace/run.sh"]
