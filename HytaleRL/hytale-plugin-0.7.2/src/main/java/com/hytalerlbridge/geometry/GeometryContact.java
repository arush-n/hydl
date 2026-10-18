package com.hytalerlbridge.geometry;

/** Authoritative block-contact result captured after the native physics tick. */
public record GeometryContact(
    int dx,
    int dy,
    int dz,
    int detailBoxIndex,
    double normalX,
    double normalY,
    double normalZ,
    double pointX,
    double pointY,
    double pointZ,
    double collisionStart,
    double collisionEnd,
    boolean touching,
    boolean overlapping
) {}
