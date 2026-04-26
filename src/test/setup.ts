import '@testing-library/jest-dom/vitest'

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverStub as typeof ResizeObserver

class DOMMatrixReadOnlyStub {
  readonly m22: number

  constructor(transform?: string) {
    const matrixValues = transform?.match(/matrix\(([^)]+)\)/)?.[1]?.split(',').map(Number)
    this.m22 = matrixValues?.[3] ?? 1
  }
}

globalThis.DOMMatrixReadOnly = DOMMatrixReadOnlyStub as typeof DOMMatrixReadOnly
