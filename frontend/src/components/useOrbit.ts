/*
 * Orbit controls for the two hand-drawn 3D canvases (the AlphaFold backbone and the protein-space
 * map): drag or arrow keys to rotate, wheel or +/- to zoom, 0 to reset, and an idle spin that
 * stops at the first interaction and never starts when the viewer asks for reduced motion.
 * The canvas backing store follows its displayed size and the device pixel ratio.
 */
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type MutableRefObject,
  type PointerEvent,
  type RefObject,
} from "react";
import { PITCH_LIMIT, ZOOM_MAX, ZOOM_MIN, clamp, type View } from "../services/structureMath";

const SPIN_STEP = 0.004;
const KEY_STEP = 0.12;
const ZOOM_STEP = 1.15;

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export interface Orbit {
  spin: boolean;
  setSpin: (on: boolean) => void;
  zoomBy: (factor: number) => void;
  reset: () => void;
  /** Spread onto the canvas. */
  handlers: {
    onKeyDown: (e: KeyboardEvent<HTMLCanvasElement>) => void;
    onPointerDown: (e: PointerEvent<HTMLCanvasElement>) => void;
    onPointerMove: (e: PointerEvent<HTMLCanvasElement>) => void;
    onPointerUp: (e: PointerEvent<HTMLCanvasElement>) => void;
    onPointerCancel: (e: PointerEvent<HTMLCanvasElement>) => void;
  };
  /** True when the last pointer press rotated the view, so its click is not a selection. */
  wasDrag: () => boolean;
}

/**
 * `viewRef` is owned by the caller so its paint function can read the view; this hook mutates it.
 */
export function useOrbit(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  viewRef: MutableRefObject<View>,
  redraw: () => void,
  start: View,
  onHover?: (x: number, y: number) => void,
): Orbit {
  const dragRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const lastDragRef = useRef(false);
  const redrawRef = useRef(redraw);
  redrawRef.current = redraw;
  const [spin, setSpin] = useState(() => !prefersReducedMotion());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const fit = (): void => {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(canvas.clientWidth * ratio);
      canvas.height = Math.round(canvas.clientHeight * ratio);
      redrawRef.current();
    };
    fit();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(fit);
    observer.observe(canvas);
    return (): void => {
      observer.disconnect();
    };
  }, [canvasRef]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  useEffect(() => {
    if (!spin) return undefined;
    let frame = 0;
    const tick = (): void => {
      viewRef.current.yaw += SPIN_STEP;
      redrawRef.current();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return (): void => {
      cancelAnimationFrame(frame);
    };
  }, [spin, viewRef]);

  // Wheel zoom needs a non-passive listener so the page does not scroll with it.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      const view = viewRef.current;
      view.zoom = clamp(view.zoom * Math.exp(-e.deltaY * 0.001), ZOOM_MIN, ZOOM_MAX);
      redrawRef.current();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return (): void => {
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [canvasRef, viewRef]);

  const rotateBy = useCallback(
    (dyaw: number, dpitch: number): void => {
      const view = viewRef.current;
      view.yaw += dyaw;
      view.pitch = clamp(view.pitch + dpitch, -PITCH_LIMIT, PITCH_LIMIT);
      redrawRef.current();
    },
    [viewRef],
  );

  const zoomBy = useCallback(
    (factor: number): void => {
      const view = viewRef.current;
      view.zoom = clamp(view.zoom * factor, ZOOM_MIN, ZOOM_MAX);
      redrawRef.current();
    },
    [viewRef],
  );

  const reset = useCallback((): void => {
    Object.assign(viewRef.current, start);
    redrawRef.current();
  }, [start, viewRef]);

  const onKeyDown = (e: KeyboardEvent<HTMLCanvasElement>): void => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const keys: Record<string, () => void> = {
      ArrowLeft: () => {
        rotateBy(-KEY_STEP, 0);
      },
      ArrowRight: () => {
        rotateBy(KEY_STEP, 0);
      },
      ArrowUp: () => {
        rotateBy(0, -KEY_STEP);
      },
      ArrowDown: () => {
        rotateBy(0, KEY_STEP);
      },
      "+": () => {
        zoomBy(ZOOM_STEP);
      },
      "=": () => {
        zoomBy(ZOOM_STEP);
      },
      "-": () => {
        zoomBy(1 / ZOOM_STEP);
      },
      "0": reset,
    };
    const action = keys[e.key];
    if (!action) return;
    e.preventDefault();
    setSpin(false);
    action();
  };

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>): void => {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY, moved: false };
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>): void => {
    const drag = dragRef.current;
    if (drag) {
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 0) {
        if (!drag.moved) setSpin(false);
        rotateBy(dx * 0.01, dy * 0.01);
        dragRef.current = { x: e.clientX, y: e.clientY, moved: true };
      }
      return;
    }
    if (onHover) {
      const rect = e.currentTarget.getBoundingClientRect();
      onHover(e.clientX - rect.left, e.clientY - rect.top);
    }
  };

  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>): void => {
    lastDragRef.current = dragRef.current?.moved === true;
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  return {
    spin,
    setSpin,
    zoomBy,
    reset,
    handlers: {
      onKeyDown,
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel: onPointerUp,
    },
    wasDrag: () => lastDragRef.current,
  };
}
