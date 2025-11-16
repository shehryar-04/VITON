import Link from "next/link";
import { Facebook, Instagram, Twitter, Youtube } from "lucide-react";

const Footer = () => {
  return (    <footer className="bg-gray-900 mt-20 text-gray-100">
      <div className="mx-auto max-w-7xl px-4 sm:px-4 lg:px-8">
        <div className="py-20 grid grid-cols-1 md:grid-cols-4 gap-12">
          {/* Quick Links */}
          <div>            <h3 className="text-white text-xl font-semibold mb-8">Quick Links</h3>
            <ul className="space-y-3">
              <li>
                <Link href="/" className="text-gray-300 hover:text-white transition">
                  Home
                </Link>
              </li>
              <li>                <Link href="/categories" className="text-gray-300 hover:text-white transition">
                  Products
                </Link>
              </li>              <li>
                <Link href="/try-history" className="text-gray-300 hover:text-white transition">
                  Try-On History
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-gray-300 hover:text-white transition">
                  About
                </Link>
              </li>
            </ul>
          </div>          {/* Categories */}
          <div>            <h3 className="text-white text-xl font-semibold mb-8">Products</h3>
            <ul className="space-y-3">
              <li>
                <Link href="/categories/t-shirts" className="text-gray-300 hover:text-white transition">
                  T-Shirts
                </Link>
              </li>
              <li>
                <Link href="/categories/pants" className="text-muted-foreground hover:text-primary transition">
                  Pants
                </Link>
              </li>
              <li>
                <Link href="/categories/jeans" className="text-muted-foreground hover:text-primary transition">
                  Jeans
                </Link>
              </li>
              <li>
                <Link href="/categories/shorts" className="text-muted-foreground hover:text-primary transition">
                  Shorts
                </Link>
              </li>
            </ul>
          </div>          {/* Contact Info */}
          <div>            <h3 className="text-white text-xl font-semibold mb-8">Contact Us</h3>
            <ul className="space-y-3">
              <li className="text-gray-300">
                Email: support@styleshift.com
              </li>
              <li className="text-gray-300">
                Phone: +92 (555) 123-4567
              </li>
              <li className="text-muted-foreground">
                Address: Commercial Area Bahawalpur Punjab, Pakistan
              </li>
            </ul>
          </div>          {/* Social Media & Newsletter */}
          <div>            <h3 className="text-white text-xl font-semibold mb-8">Connect With Us</h3>
            <div className="flex space-x-6 mb-8">
              <a href="#" className="text-gray-300 hover:text-white transition">
                <Facebook className="h-6 w-6" />
              </a>
              <a href="#" className="text-muted-foreground hover:text-primary transition">
                <Instagram className="h-6 w-6" />
              </a>
              <a href="#" className="text-muted-foreground hover:text-primary transition">
                <Twitter className="h-6 w-6" />
              </a>
              <a href="#" className="text-muted-foreground hover:text-primary transition">
                <Youtube className="h-6 w-6" />
              </a>
            </div>
            <div>
              <h4 className="text-sm font-semibold mb-3">Subscribe to our newsletter</h4>
              <div className="flex gap-2">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="flex-1 px-3 py-2 border rounded-md text-sm"
                />
                <button
                  className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm hover:opacity-90 transition"
                >
                  Subscribe
                </button>
              </div>
            </div>
          </div>
        </div>        {/* Bottom Bar */}
        <div className="border-t py-5">
          <div className="flex flex-col md:flex-row justify-between items-center mt-2">
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} StyleShift. All rights reserved.
            </p>
            <div className="flex space-x-8 mt-6 md:mt-0">
              <Link href="#" className="text-sm text-muted-foreground hover:text-primary transition">
                Privacy Policy
              </Link>
              <Link href="#" className="text-sm text-muted-foreground hover:text-primary transition">
                Terms of Service
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
