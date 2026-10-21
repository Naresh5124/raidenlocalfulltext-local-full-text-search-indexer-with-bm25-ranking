# ADR-0004: Authentication and Session Management

## Status
Accepted

## Context
User identity must be validated safely across API routes, websockets, and background tasks.

## Decision
Adopt bearer token and session-based identity verification with cryptographic signatures and salted credentials.

## Consequences
Predictable access control and clean separation between unauthenticated and secured routes.
