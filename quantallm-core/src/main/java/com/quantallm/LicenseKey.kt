package com.quantallm

import android.content.Context
import android.util.Base64
import org.json.JSONObject
import java.nio.charset.StandardCharsets
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

class LicenseKey private constructor(
    val packageName: String,
    val expiryEpoch: Long,
    val issuedAt: Long,
    val tier: String
) {

    fun isExpired(): Boolean = System.currentTimeMillis() / 1000 > expiryEpoch

    val expiryDate: String
        get() {
            val sdf = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US)
            sdf.timeZone = java.util.TimeZone.getTimeZone("UTC")
            return sdf.format(java.util.Date(expiryEpoch * 1000))
        }

    companion object {
        private const val HMAC_ALGO = "HmacSHA256"

        private val SECRET_HASH = byteArrayOf(
            0x71, 0x75, 0x61, 0x6e, 0x74, 0x61, 0x6c, 0x6c,
            0x6d, 0x2d, 0x73, 0x64, 0x6b, 0x2d, 0x76, 0x31
        )
        private val SECRET_SALT = byteArrayOf(
            0x51, 0x4c, 0x4d, 0x2d, 0x32, 0x30, 0x32, 0x36,
            0x2d, 0x6c, 0x69, 0x63, 0x65, 0x6e, 0x73, 0x65
        )

        private fun deriveSecret(): ByteArray {
            val combined = ByteArray(SECRET_HASH.size + SECRET_SALT.size)
            System.arraycopy(SECRET_HASH, 0, combined, 0, SECRET_HASH.size)
            System.arraycopy(SECRET_SALT, 0, combined, SECRET_HASH.size, SECRET_SALT.size)
            val md = java.security.MessageDigest.getInstance("SHA-256")
            return md.digest(combined)
        }

        fun validate(context: Context, token: String): LicenseKey {
            val parts = token.trim().split(".")
            if (parts.size != 3) {
                throw LicenseException("Invalid license key format")
            }

            val headerB64 = parts[0]
            val payloadB64 = parts[1]
            val signatureB64 = parts[2]

            val secret = deriveSecret()
            val signInput = "$headerB64.$payloadB64"
            val mac = Mac.getInstance(HMAC_ALGO)
            mac.init(SecretKeySpec(secret, HMAC_ALGO))
            val expectedSig = mac.doFinal(signInput.toByteArray(StandardCharsets.UTF_8))
            val actualSig = base64UrlDecode(signatureB64)

            if (!expectedSig.contentEquals(actualSig)) {
                throw LicenseException("Invalid license key signature")
            }

            val payloadJson = String(base64UrlDecode(payloadB64), StandardCharsets.UTF_8)
            val payload = try {
                JSONObject(payloadJson)
            } catch (e: Exception) {
                throw LicenseException("Malformed license key payload")
            }

            val pkg = payload.optString("pkg", "")
            val exp = payload.optLong("exp", 0)
            val iat = payload.optLong("iat", 0)
            val tier = payload.optString("tier", "standard")

            if (pkg.isEmpty()) {
                throw LicenseException("License key missing package name")
            }

            val appPkg = context.packageName
            if (pkg != appPkg) {
                throw LicenseException("License key is for package '$pkg', but this app is '$appPkg'")
            }

            if (exp <= 0) {
                throw LicenseException("License key missing expiry")
            }

            val now = System.currentTimeMillis() / 1000
            if (now > exp) {
                val key = LicenseKey(pkg, exp, iat, tier)
                throw LicenseException("License expired on ${key.expiryDate}")
            }

            return LicenseKey(pkg, exp, iat, tier)
        }

        private fun base64UrlDecode(input: String): ByteArray {
            val padded = when (input.length % 4) {
                2 -> "$input=="
                3 -> "$input="
                else -> input
            }
            return Base64.decode(
                padded.replace('-', '+').replace('_', '/'),
                Base64.DEFAULT
            )
        }
    }
}
