// Phones get a notice instead of the game: on a phone-sized screen the HUD
// covers the rocket and the touch controls take the rest. Tablets (bigger
// touch screens) keep the touch controls.

/**
 * Shortest screen side, in CSS pixels, below which a touch-only device counts
 * as a phone (Android's own phone/tablet boundary). Phones are about 360–520
 * even at small display sizes; small tablets and open foldables are 600 or more.
 */
const PHONE_MAX_SHORT_SIDE = 600;

/** Whether to show the phone notice: a touch-only device (no hover, coarse pointer) with a phone-sized screen. */
export function isPhone(screenWidth: number, screenHeight: number, touchOnly: boolean): boolean {
  return touchOnly && Math.min(screenWidth, screenHeight) < PHONE_MAX_SHORT_SIDE;
}
